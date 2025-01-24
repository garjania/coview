import timm

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
            sorted_indices = torch.argsort(frame_id)
            embeddings = embeddings[sorted_indices]
        return embeddings
        
    def get_similarity_matrix(self, embeddings, temperature):
        # Normalize the embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute the similarity matrix and scale it by the temperature
        similarity_matrix = torch.mm(embeddings[:-2], embeddings.t()[:,:-1])
        similarity_matrix /= temperature
        
        # Create a mask for the lower triangular part and set the lower triangular part to -inf
        mask = torch.tril(torch.ones_like(similarity_matrix), diagonal=-1)
        similarity_matrix = similarity_matrix.masked_fill(mask.bool(), float('-inf'))
        
        return similarity_matrix
    
    def get_softmax_similarity_matrix(self, x, frame_id):
        embeddings = self.backbone(x)
        embeddings = self._sort_embeddings(embeddings, frame_id)
        similarity_matrix = self.get_similarity_matrix(embeddings, self.temperature)
        return F.softmax(similarity_matrix, dim=1)
        
    def forward_loss(self, embeddings, frame_id):
        embeddings = self._sort_embeddings(embeddings, frame_id)
        similarity_matrix = self.get_similarity_matrix(embeddings, self.temperature)
        # Compute cross-entropy loss
        labels = torch.arange(similarity_matrix.size(0)).to(similarity_matrix.device)
        return F.cross_entropy(similarity_matrix, labels)
        
    def forward(self, x, frame_id=None, return_loss=False):
        if return_loss:
            embeddings = self.backbone(x)
            loss = self.forward_loss(embeddings, frame_id)
            return embeddings, loss
        else:
            return self.backbone(x)
    
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

        self.backbone.forward_features(image)

        hook_handle.remove()

        if not attention_scores:
            raise ValueError("Attention scores were not captured")
        
        attention_scores = attention_scores[0]
        cls_attention = attention_scores[:, :, 0, :]
        cls_to_image_attention = cls_attention[:, :, 1:]
        
        return cls_to_image_attention