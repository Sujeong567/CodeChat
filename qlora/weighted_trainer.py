# weighted_trainer.py
import torch
from trl import SFTTrainer

class WeightedSFTTrainer(SFTTrainer):
    def __init__(self, *args, review_weight=3.0, review_weight_fn=None, log_review_weight_steps=200, **kwargs):
        super().__init__(*args, **kwargs)
        self.review_weight = review_weight
        self.review_weight_fn = review_weight_fn
        self.log_review_weight_steps = log_review_weight_steps

        # 특수 토큰 ID
        self.review_start_id = self.tokenizer.convert_tokens_to_ids("<REVIEW_START>")
        self.review_end_id   = self.tokenizer.convert_tokens_to_ids("<REVIEW_END>")

        # 어시스턴트 태그 인코딩(개행 포함 주의: 학습 템플릿과 반드시 동일)
        self.assistant_ids = self.tokenizer.encode("<|assistant|>\n", add_special_tokens=False)

        if self.review_start_id == -1 or self.review_end_id == -1:
            raise ValueError("'<REVIEW_START>' / '<REVIEW_END>' must be added as additional_special_tokens.")

    def _find_subseq(self, seq, subseq):
        Ls, Lp = len(seq), len(subseq)
        if Lp == 0 or Lp > Ls:
            return -1
        for i in range(Ls - Lp + 1):
            if seq[i:i+Lp] == subseq:
                return i
        return -1

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # base labels
        if "labels" in inputs and inputs["labels"] is not None:
            labels = inputs["labels"].clone()
        else:
            labels = inputs["input_ids"].clone()

        input_ids = inputs["input_ids"]
        attn_mask = inputs.get("attention_mask", None)
        outputs = model(**inputs)
        logits = outputs.logits
        B, T, V = logits.shape

        # 기본 weight=1
        weights = torch.ones_like(labels, dtype=torch.float)

        # 스케줄된 리뷰 가중치
        review_w = self.review_weight
        if self.review_weight_fn is not None and model.training:
            review_w = float(self.review_weight_fn(self.state.global_step))

        # (0) 패딩 마스킹: attention_mask==0인 곳은 학습 제외
        if attn_mask is not None:
            pad_pos = (attn_mask == 0)
            labels[pad_pos] = -100
            weights[pad_pos] = 0.0

        # (1) <|assistant|>\n 이전은 전부 무시
        for b in range(B):
            ids = input_ids[b].tolist()
            pos = self._find_subseq(ids, self.assistant_ids)
            if pos == -1:
                labels[b, :] = -100
                weights[b, :] = 0.0
                continue
            upto = pos + len(self.assistant_ids)
            labels[b, :upto] = -100
            weights[b, :upto] = 0.0

        # (2) 리뷰 구간 가중치 적용 (마커 자체는 라벨 제외)
        for b in range(B):
            in_review = False
            for t in range(T):
                tok = labels[b, t].item()

                if tok == self.review_start_id:
                    labels[b, t] = -100
                    weights[b, t] = 0.0
                    in_review = True
                    continue

                if tok == self.review_end_id:
                    labels[b, t] = -100
                    weights[b, t] = 0.0
                    in_review = False
                    continue

                if in_review and labels[b, t] != -100:
                    weights[b, t] = review_w

        # (3) causal shift 적용
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_weights = weights[:, 1:].contiguous()

        # 손실 계산 (-100 무시)
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        per_tok_loss = loss_fct(
            shift_logits.view(-1, V),
            shift_labels.view(-1)
        ).view(B, T - 1)

        # 유효 토큰에 대해서만 가중 평균
        valid_mask = (shift_labels != -100) & (shift_weights != 0)
        denom = valid_mask.sum().clamp(min=1)
        weighted_loss = (per_tok_loss * shift_weights).sum() / denom

        # 디버그 로그
        if model.training and self.state.global_step % self.log_review_weight_steps == 0:
            valid_label_tokens = (shift_labels != -100).sum().item()
            print(
                f"[Step {self.state.global_step}] review_weight = {review_w:.4f}, "
                f"valid_label_tokens = {valid_label_tokens}"
            )

        return (weighted_loss, outputs) if return_outputs else weighted_loss
