import librosa
import soundfile as sf
import torch
from ns3_codec import FACodecEncoderV2, FACodecDecoderV2
from huggingface_hub import hf_hub_download
from ns3_codec import FACodecEncoder, FACodecDecoder


def load_audio(wav_path):
  wav = librosa.load(wav_path, sr=16000)[0]
  wav = torch.from_numpy(wav).float()
  wav = wav.unsqueeze(0).unsqueeze(0)
  return wav


fa_encoder = FACodecEncoder(
    ngf=32,
    up_ratios=[2, 4, 5, 5],
    out_channels=256,
)

fa_decoder = FACodecDecoder(
    in_channels=256,
    upsample_initial_channel=1024,
    ngf=32,
    up_ratios=[5, 5, 4, 2],
    vq_num_q_c=2,
    vq_num_q_p=1,
    vq_num_q_r=3,
    vq_dim=256,
    codebook_dim=8,
    codebook_size_prosody=10,
    codebook_size_content=10,
    codebook_size_residual=10,
    use_gr_x_timbre=True,
    use_gr_residual_f0=True,
    use_gr_residual_phone=True,
)

encoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_encoder.bin")
decoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_decoder.bin")

fa_encoder.load_state_dict(torch.load(encoder_ckpt))
fa_decoder.load_state_dict(torch.load(decoder_ckpt))

fa_encoder.eval()
fa_decoder.eval()

with torch.no_grad():
    test_wav = load_audio("./audio/1.wav")

    # encode
    enc_out = fa_encoder(test_wav)
    print(enc_out.shape)

    # quantize
    vq_post_emb, vq_id, _, quantized, spk_embs = fa_decoder(enc_out, eval_vq=False, vq=True)
    
    # latent after quantization
    print(vq_post_emb.shape)
    
    # codes
    print("vq id shape:", vq_id.shape)
    
    # get prosody code
    prosody_code = vq_id[:1]
    print("prosody code shape:", prosody_code.shape)
    # print("prosody code:", prosody_code)
    
    # get content code
    cotent_code = vq_id[1:3]
    print("content code shape:", cotent_code.shape)
    # print("content code:", cotent_code)
    
    # get residual code (acoustic detail codes)
    residual_code = vq_id[3:]
    print("residual code shape:", residual_code.shape)
    # print("residual code:", residual_code)
    
    # speaker embedding
    print("speaker embedding shape:", spk_embs.shape)
    # print("speaker embedding:", spk_embs)

    # decode (recommand)
    recon_wav = fa_decoder.inference(vq_post_emb, spk_embs)
    print(recon_wav.shape)
    sf.write("./audio/1_recon.wav", recon_wav[0][0].cpu().numpy(), 16000)

fa_encoder_v2 = FACodecEncoderV2(
    ngf=32,
    up_ratios=[2, 4, 5, 5],
    out_channels=256,
)

fa_decoder_v2 = FACodecDecoderV2(
    in_channels=256,
    upsample_initial_channel=1024,
    ngf=32,
    up_ratios=[5, 5, 4, 2],
    vq_num_q_c=2,
    vq_num_q_p=1,
    vq_num_q_r=3,
    vq_dim=256,
    codebook_dim=8,
    codebook_size_prosody=10,
    codebook_size_content=10,
    codebook_size_residual=10,
    use_gr_x_timbre=True,
    use_gr_residual_f0=True,
    use_gr_residual_phone=True,
)

encoder_v2_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_encoder_v2.bin")
decoder_v2_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_decoder_v2.bin")

fa_encoder_v2.load_state_dict(torch.load(encoder_v2_ckpt))
fa_decoder_v2.load_state_dict(torch.load(decoder_v2_ckpt))


with torch.no_grad():
  wav_a = load_audio("./audio/1.wav")
  wav_b = load_audio("./audio/2.wav")
  enc_out_a = fa_encoder_v2(wav_a)
  prosody_a = fa_encoder_v2.get_prosody_feature(wav_a)
  enc_out_b = fa_encoder_v2(wav_b)
  prosody_b = fa_encoder_v2.get_prosody_feature(wav_b)

  vq_post_emb_a, vq_id_a, _, quantized, spk_embs_a = fa_decoder_v2(
      enc_out_a, prosody_a, eval_vq=False, vq=True
  )
  vq_post_emb_b, vq_id_b, _, quantized, spk_embs_b = fa_decoder_v2(
      enc_out_b, prosody_b, eval_vq=False, vq=True
  )

  vq_post_emb_a_to_b = fa_decoder_v2.vq2emb(vq_id_a, use_residual=False)
  recon_wav_a_to_b = fa_decoder_v2.inference(vq_post_emb_a_to_b, spk_embs_b)
  sf.write("/Users/feiteng/speech/ns3_codec/audio/1_to_2.wav", recon_wav_a_to_b[0][0].cpu().numpy(), 16000)
