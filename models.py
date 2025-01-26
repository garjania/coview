import timm
from einops import rearrange, repeat

import torch
from torch import nn
from torch.nn import functional as F

class FrameEncoder(nn.Module):
    def __init__(self, 
                 name='vit_base_patch16_224', 
                 temperature=0.05, 
                 pretrained=False,
                ):
        super().__init__()
        # Load ViT base model
        self.backbone = timm.create_model(
            name,
            pretrained=pretrained,
            num_classes=0  # Remove classification head
        )
        # Get embedding dimension from the model
        self.embedding_dim = self.backbone.embed_dim
        self.temperature = temperature
        
    def _sort_embeddings(self, embeddings, frame_id):
        # Sort the embeddings by frame_id
        if frame_id is not None:
            # Get sorted indices for all batches at once
            _, sorted_indices = torch.sort(frame_id, dim=1)
            # Expand sorted_indices to match embeddings dimensions
            sorted_indices = repeat(sorted_indices, 'b n -> b n d', d=embeddings.size(-1))
            # Use gather to sort embeddings
            embeddings = torch.gather(embeddings, dim=1, index=sorted_indices)
        return embeddings
        
    def get_similarity_matrix(self, embeddings, temperature):
        # Normalize the embeddings
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        # Compute the similarity matrix and scale it by the temperature
        similarity_matrix = torch.bmm(
            embeddings[:,:-2], 
            embeddings[:,:-1].transpose(1, 2)
        )
        similarity_matrix /= temperature
        # Create a mask for the lower triangular part and set the lower triangular part to -inf
        mask = torch.tril(torch.ones_like(similarity_matrix), diagonal=-1)
        similarity_matrix = similarity_matrix.masked_fill(mask.bool(), float('-inf'))
        return similarity_matrix
    
    def get_softmax_similarity_matrix(self, x, frame_id):
        embeddings = self.forward_backbone(x)
        embeddings = self._sort_embeddings(embeddings, frame_id)
        similarity_matrix = self.get_similarity_matrix(embeddings, self.temperature)
        return F.softmax(similarity_matrix, dim=1)
    
    def forward_backbone(self, x, visual_features=False):
        reshaped = False
        if x.ndim == 5:
            B = x.shape[0]
            x = rearrange(x, 'b n c h w -> (b n) c h w')
            reshaped = True
        if visual_features:
             features = self.backbone.forward_features(x)
        else:
            features = self.backbone(x)
        if reshaped:
            features = rearrange(features, '(b n) c -> b n c', b=B)
        return features
    
    def forward_loss(self, embeddings, frame_id):
        embeddings = self._sort_embeddings(embeddings, frame_id)
        similarity_matrix_1 = self.get_similarity_matrix(embeddings, self.temperature)
        similarity_matrix_2 = self.get_similarity_matrix(embeddings.flip(dims=(1,)), self.temperature)
        similarity_matrix = (similarity_matrix_1 + similarity_matrix_2) / 2
        # Create labels for cross-entropy loss
        labels = torch.arange(similarity_matrix.size(1)).to(similarity_matrix.device)
        labels = repeat(labels, 'n -> b n', b=similarity_matrix.size(0))
        # Reshape similarity matrix and labels to be compatible with cross-entropy loss
        similarity_matrix = similarity_matrix.reshape(-1, similarity_matrix.size(-1))
        labels = labels.reshape(-1)
        return F.cross_entropy(similarity_matrix, labels, reduction='mean')
        
    def forward(self, x, frame_id=None, return_loss=False):
        if return_loss:
            embeddings = self.forward_backbone(x)
            loss = self.forward_loss(embeddings, frame_id)
            return embeddings, loss
        else:
            return self.forward_backbone(x)
    
    @torch.no_grad()
    def get_cls_attention_from_last_layer(self, image):
        attention_scores = []

        def hook_fn(module, input, output):
            x = input[0]
            B, N, C = x.shape
            qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, module.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q, k = module.q_norm(q), module.k_norm(k)
            
            q = q * module.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            
            attention_scores.append(attn)

        # Register the hook on the attention block itself
        last_block = self.backbone.blocks[-1].attn
        hook_handle = last_block.register_forward_hook(hook_fn)

        self.forward_backbone(image, visual_features=True)

        hook_handle.remove()

        if not attention_scores:
            raise ValueError("Attention scores were not captured")
        
        attention_scores = attention_scores[0]
        cls_attention = attention_scores[:, :, 0, :]
        cls_to_image_attention = cls_attention[:, :, 1:]
        
        return cls_to_image_attention