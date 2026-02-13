# ComfyI2I (Modernized, Inpaint-Focused)

Fork modernizado do `ManglerFTW/ComfyI2I`, redesenhado para inpaint de alta qualidade em ComfyUI atual.

Objetivos desta versão:
- qualidade máxima de crop/resize/paste (incluindo `lanczos` real)
- robustez para batch/lista e mapeamentos de máscara
- evitar crashes/OOM por índices ou dimensões inválidas
- fluxo focado em máscara manual/externa (sem auto-mask por texto/CLIP)

## O que mudou

### Modernização dos nodes principais

1. `Mask Ops`
- removido caminho de máscara automática por texto/CLIP
- pipeline robusto de levels + grow/erode + blur + threshold
- separação por componentes conectados com controles:
  - `component_min_area`
  - `max_components`
- `blend_percentage` agora com modos:
  - `global` (antigo comportamento)
  - `edge_band` (blend só ao redor da máscara)
- saídas estáveis para batch e `MASK_MAPPING`

2. `Color Transfer`
- modos de match de cor:
  - `lab_reinhard`
  - `rgb_stats`
- opção `preserve_luminance`
- suporta source/target com resoluções diferentes
- suporte consistente a máscara e batch
- quantização opcional por paleta (`no_of_colors`)

3. `Inpaint Segments`
- proteção contra dimensões perigosas e explosão de memória
- `resize_method` com `lanczos`
- comportamento para máscara vazia:
  - `full_image`
  - `skip`
- controle `max_output_regions` (limite de segurança de regiões)
- saída extra de `crop mapping` para pipelines avançados batch/lista

4. `Combine and Paste`
- colagem com melhor qualidade e menos artefato
- `op` agora inclui `mask_blend` (modo recomendado para inpaint)
- opções novas:
  - `patch_resize_method`
  - `mask_resize_method`
  - `edge_fix_strength`
  - `detail_boost`
  - `post_sharpen`
  - `max_regions`
- `max_regions`: limite de segurança para evitar processar regiões demais e travar fluxo

### Novo node de alinhamento automático

5. `I2I Auto Align Image+Mask`
- prepara imagem e máscara para dimensões corretas automaticamente
- modos de target:
  - `use_image_size`
  - `use_mask_size`
  - `custom`
- modos de ajuste:
  - `stretch`
  - `contain`
  - `cover`
- âncoras e `auto_multiple_of` para manter compatibilidade com pipelines de inpaint

### Outros nodes utilitários avançados

- `I2I Mask Refiner Pro`
- `I2I Masked Tile Extractor`
- `I2I Region Overlay Debug`
- `I2I Seamless Patch Paste`
- `I2I Detail Preserve Blend`
  - agora com `output_size` + `resize_method` para evitar erro de broadcast em imagens de tamanhos diferentes

## Instalação

```bash
git clone https://github.com/adbrasi/ComfyI2I.git
cd ComfyI2I
```

Reinicie o ComfyUI.

## Dependências

As dependências não estão mais declaradas em `requirements.txt` neste fork, conforme solicitado.
Use o ambiente/deploy que você já mantém para instalar/gerenciar os pacotes necessários.

## Observações

- `install.bat` foi removido.
- fluxo de auto-mask por texto/CLIP foi removido para simplificar, modernizar e reduzir fragilidade.
- foco total do pacote: administração de máscaras, regiões, tamanhos, qualidade de colagem e tratamento robusto em inpaint.

## Licença

MIT (mesma licença do upstream).
