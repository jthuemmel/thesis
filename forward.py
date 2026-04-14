
# MAE TOK
# def forward(self, fields: torch.FloatTensor, visible: torch.BoolTensor) -> torch.FloatTensor:
#         B = fields.size(0)
#         # tokenize and add position codes
#         tokens = self.to_tokens(fields) + self.src_positions

#         # select visible
#         src = einops.rearrange(tokens[visible], '(b m) ... -> b m ...', b = B)
        
#         # latent encoder
#         latents = einops.repeat(self.latent_tokens, 'z d -> b z d', b = B)
#         latents, shape = einops.pack([src, latents], 'b * d')
#         for read in self.encoder:
#             latents = read(latents)

#         # project latents to decoder
#         _, latents = einops.unpack(latents, shape, 'b * d')
#         latents = self.to_decoder(latents)

#         # create queries from mask tokens and scatter src tokens
#         tgt = einops.repeat(self.mask_token, 'd -> b n d', b = B, n = tokens.size(1))
#         tgt = tgt + self.tgt_positions

#         # decoder
#         tgt, shape = einops.pack([tgt, latents], 'b * d')
#         for write in self.decoder:
#             tgt = write(tgt)

#         # prediction head
#         tgt, _ = einops.unpack(latents, shape, 'b * d')
#         pred = self.to_output(tgt)
#         return pred


# PERCEIVER
# def forward(self, fields: torch.FloatTensor, visible: torch.BoolTensor) -> torch.FloatTensor:
#         B = fields.size(0)
#         # tokenize and add position codes
#         tokens = self.to_tokens(fields) + self.src_positions

#         # select visible
#         src = einops.rearrange(tokens[visible], '(b m) ... -> b m ...', b = B)
        
#         # latent encoder
#         latents = einops.repeat(self.latent_tokens, 'z d -> b z d', b = B)
#         for i, read in enumerate(self.encoder):
#             latents = read(latents, src) if i == 0 else read(latents)

#         latents = self.to_decoder(latents)

#         # create queries from mask tokens and scatter src tokens
#         tgt = einops.repeat(self.mask_token, 'd -> b n d', b = B, n = tokens.size(1))
#         tgt = tgt + self.tgt_positions

#         # decoder
#         for write in self.decoder:
#             tgt = write(tgt, latents)

#         # prediction head
#         pred = self.to_output(tgt)
#         return pred

# FLAMINGO
# def forward(self, fields: torch.FloatTensor, visible: torch.BoolTensor) -> torch.FloatTensor:
#         B = fields.size(0)
#         # tokenize and add position codes
#         tokens = self.to_tokens(fields) + self.src_positions

#         # select visible
#         src = einops.rearrange(tokens[visible], '(b m) ... -> b m ...', b = B)
        
#         # latent encoder
#         latents = einops.repeat(self.latent_tokens, 'z d -> b z d', b = B)
#         for i, read in enumerate(self.encoder):
#             latents = read(latents, torch.cat([src, latents], dim = 1))

#         latents = self.to_decoder(latents)

#         # create queries from mask tokens and scatter src tokens
#         tgt = einops.repeat(self.mask_token, 'd -> b n d', b = B, n = tokens.size(1))
#         tgt = tgt + self.tgt_positions

#         # decoder
#         for write in self.decoder:
#             tgt = write(tgt, torch.cat([latents, tgt], dim = 1))

#         # prediction head
#         pred = self.to_output(tgt)
#         return pred

# VIT
    # def forward(self, fields: torch.FloatTensor, visible: torch.BoolTensor) -> torch.FloatTensor:
    #     # tokenize and add position codes
    #     tokens = self.to_tokens(fields) + self.src_positions

    #     # select visible
    #     src = einops.rearrange(tokens[visible], '(b m) ... -> b m ...', b = tokens.size(0))

    #     # encoder
    #     for read in self.encoder:
    #         src = read(src)
    #     src = self.to_decoder(src)

    #     # create queries from mask tokens and scatter src tokens
    #     tgt = einops.repeat(self.mask_token.type_as(src), 'd -> b n d', b = tokens.size(0), n = tokens.size(1))
    #     mask = einops.repeat(visible, 'b m -> b m d', d = tgt.size(-1))
    #     tgt = tgt.masked_scatter(
    #         mask = mask,
    #         source = src
    #     )
    #     tgt = tgt + self.tgt_positions

    #     # decoder
    #     for write in self.decoder:
    #         tgt = write(tgt)

    #     # prediction head
    #     pred = self.to_output(tgt)
    #     return pred