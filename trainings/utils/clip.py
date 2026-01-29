import torch


def get_embds_from_3_clip(embd1, embd2, embd3):
    clips_embds = torch.cat((embd1, embd2), dim=-1)  # [1, 77, 2048]

    clips_embds = torch.nn.functional.pad(
        clips_embds, (0, embd3.shape[-1] - clips_embds.shape[-1])
    )  # [1, 77, 4096]

    clips_embds = torch.cat((embd3, clips_embds), dim=-2)  # [1, 333, 4096]

    return clips_embds


def get_3_clip_from_embds(
    clips_embds,
    encoder_1_seq,
    encoder_1_channels,
    encoder_2_channels,
):
    t5_embeds = clips_embds[:, :-encoder_1_seq, :]
    clips_embds = clips_embds[:, -encoder_1_seq:, :]

    clips_embds_1 = clips_embds[:, :, :encoder_1_channels]
    clips_embds_2 = clips_embds[
        :, :, encoder_1_channels : encoder_1_channels + encoder_2_channels
    ]

    return clips_embds_1, clips_embds_2, t5_embeds
