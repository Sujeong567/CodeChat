# server/lora/inference.py
import tenseal as ts

def he_lora_inference(enc_input, W_A_pt, W_B_pt, ctx):
    enc_intermediate = enc_input.matmul(W_A_pt)
    enc_output = enc_intermediate.matmul(W_B_pt)
    try:
        enc_output.rescale_()
    except:
        pass
    return enc_output.serialize()


def he_lora_inference_multi(enc_input, proj_tensors, ctx):
    results = {}

    for proj, (W_A_pt, W_B_pt) in proj_tensors.items():
        results[proj] = he_lora_inference(enc_input, W_A_pt, W_B_pt, ctx)

    return results
