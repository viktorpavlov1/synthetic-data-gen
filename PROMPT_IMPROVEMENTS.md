# Prompt Engineering Improvements

## Overview

The image generation system has been significantly improved with:
1. **Professional prompt engineering** based on your successful example
2. **Diagnosis-specific prompts** for all 7 HAM10000 lesion types
3. **Controlled distribution** for mixed diagnosis generation
4. **HAM10000 format** (600x450) for all generated images

## New Prompt Structure

Based on your successful example and best practices, prompts now include:

### Core Elements:
- **Ultra photorealistic ISIC-style** - Matches HAM10000 dataset style
- **Dermoscopic close-up photograph** - Specific medical imaging type
- **Contact polarized dermatoscope** - Technical accuracy
- **Immersion fluid** - Realistic dermoscopy details
- **3:4 aspect ratio** - Matches HAM10000 format
- **Lesion positioning** - "occupying about one third of the frame"
- **Clinical details** - Body hairs, surface reflections, lighting
- **Image quality** - Razor-sharp focus, natural texture, realistic colors
- **No artistic style** - Pure clinical photography

### Diagnosis-Specific Features:

Each diagnosis type includes relevant clinical features:

- **Melanoma (mel)**: Irregular borders, asymmetric pattern, variegated color, atypical pigment network
- **Nevus (nv)**: Regular borders, uniform pigment network, symmetric pattern
- **BCC (bcc)**: Rolled borders, telangiectasias, pearly appearance
- **Vascular (vasc)**: Reddish color, lacunar structures, vascular patterns
- **BKL (bkl)**: Well-defined borders, scaly surface, uniform color
- **AKIEC (akiec)**: Scaly surface, irregular borders, erythematous base
- **DF (df)**: Central dimple, firm texture, brownish color

## Interface Features

### 1. Single Diagnosis Mode
- Select one specific diagnosis type
- Generate all images of that type
- Consistent, focused generation

### 2. Mixed Diagnoses Mode
- Select multiple diagnosis types
- Control exact distribution (percentages)
- Visual feedback on distribution sum
- Automatic equal distribution option

### 3. Image Size
- Fixed to **600x450** (HAM10000 format)
- All generated images match dataset dimensions
- Proper aspect ratio maintained

## Prompt Examples

### Melanoma:
```
ultra photorealistic ISIC-style dermoscopic close-up photograph of a single melanoma skin lesion on real human skin, taken with a contact polarized dermatoscope using immersion fluid, 3:4 aspect ratio, lesion occupying about one third of the frame on a light pink skin background, full lesion visible and slightly off-center, fine body hairs crossing the lesion and surrounding skin, subtle surface reflections from the dermatoscope contact plate, even soft clinical lighting with no harsh shadows, razor-sharp focus over the entire lesion, natural skin texture and pores clearly visible, realistic color range from light pink to dark brown and black, irregular pigment network and internal structures clearly defined, showing irregular borders, asymmetric pattern, variegated color, atypical pigment network, unedited hospital dermatology research image, no artistic style, no filters, clinical photography quality, medical grade image
```

### Melanocytic Nevus:
```
ultra photorealistic ISIC-style dermoscopic close-up photograph of a single melanocytic nevi skin lesion on real human skin, taken with a contact polarized dermatoscope using immersion fluid, 3:4 aspect ratio, lesion occupying about one third of the frame on a light pink skin background, full lesion visible and slightly off-center, fine body hairs crossing the lesion and surrounding skin, subtle surface reflections from the dermatoscope contact plate, even soft clinical lighting with no harsh shadows, razor-sharp focus over the entire lesion, natural skin texture and pores clearly visible, realistic color range from light tan to dark brown, regular pigment network and uniform internal structures, showing regular borders, uniform pigment network, symmetric pattern, unedited hospital dermatology research image, no artistic style, no filters, clinical photography quality, medical grade image
```

## Best Practices Applied

1. **Specificity**: Detailed technical descriptions
2. **Medical Accuracy**: Correct terminology and features
3. **Visual Details**: Lighting, texture, positioning
4. **Quality Markers**: "ultra photorealistic", "razor-sharp", "medical grade"
5. **Negative Prompts**: "no artistic style", "no filters"
6. **Context**: ISIC-style, hospital research image
7. **Aspect Ratio**: Explicit 3:4 ratio matching HAM10000

## Usage

1. **Select Diagnosis Mode**: Single or Mixed
2. **Choose Types**: Select from 7 HAM10000 diagnosis types
3. **Set Distribution**: Control percentages for mixed mode
4. **Generate**: Images will use improved prompts automatically

## Technical Details

- **Image Size**: 600x450 pixels (width x height)
- **Aspect Ratio**: 4:3 (matches HAM10000)
- **Prompt Length**: ~150-200 words for maximum detail
- **Generation**: All models (Stable Diffusion, QWEN, Flux) use same format

## Benefits

1. **Better Quality**: More detailed prompts = better images
2. **Clinical Accuracy**: Diagnosis-specific features
3. **Consistency**: All images match HAM10000 format
4. **Control**: Precise distribution control
5. **Realism**: Based on proven successful prompts

