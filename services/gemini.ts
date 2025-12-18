
import { GoogleGenAI, Type } from '@google/genai';
import type { AspectRatio } from '../types';

if (!process.env.API_KEY) {
    throw new Error("API_KEY environment variable not set");
}

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

const PROMPT_GENERATION_MODEL = 'gemini-2.5-flash';
const IMAGE_GENERATION_MODEL = 'imagen-4.0-generate-001';

const promptSchema = {
  type: Type.OBJECT,
  properties: {
    prompts: {
      type: Type.ARRAY,
      items: {
        type: Type.STRING,
        description: 'A single, descriptive, multi-line prompt for generating an image based on a scene from the script.',
      },
    },
  },
  required: ['prompts'],
};

export async function generatePrompts(script: string, style: string, niche: string): Promise<{ prompts: string[], requestPrompt: string }> {
    const requestPrompt = `System Instruction: You are an expert script analyst and creative director specializing in creating highly detailed, multi-layered visual scenes for documentary-style motion graphics. Your task is to transform a script into a series of structured, descriptive prompts for a text-to-image AI.

User Request:
I have a script that needs to be visualized. Please generate a series of image prompts based on it, following a very specific layered format.

**Context & Style:**
${niche ? `- **Topic/Niche:** ${niche}\n` : ''}- **Visual Style:** ${style}

**CRITICAL INSTRUCTIONS:**
1.  **Analyze and Breakdown:** Read the script and divide it into key visual moments or scenes.
2.  **Structure Each Prompt:** For each scene, create a SINGLE multi-line prompt. Each line within the prompt must describe a distinct element of the scene (e.g., Background, Character, Object, Text Overlay, Special Effect). Use newline characters to separate elements.
3.  **Detailed Element Descriptions:** For each element, clearly define its role, style, and context. The goal is to describe a highly creative documentary video scene.
4.  **Adherence to Style:** Every element's description MUST strictly adhere to the provided **Visual Style** and incorporate the **Topic/Niche** (if provided).
5.  **IMPORTANT SAFETY RULE:** You MUST generate prompts that are safe and appropriate for a general audience. Do not describe or imply violence, explicit situations, or sensitive interactions. If the script contains such themes, you MUST represent them abstractly or symbolically. Focus on setting, atmosphere, and emotion.
6.  **Output Format:** Your entire response MUST be a valid JSON object with a single key "prompts", which is an array of strings. Each string in the array is a complete, multi-line image prompt for one scene. Do not add any commentary, explanations, or markdown formatting around the JSON.

**Example of a SINGLE Prompt String (for one scene):**
Background: Complex Financial System Diagram - The abstract, intricate diagram of interconnected gears and pipes , now slightly out of focus or with reduced opacity to serve as a background element. Style: A Minimalist 2D Vector Illustration, Flat Design 2.0, with Glowing Light Effects in Electric Green on a Deep Black Canvas, designed for a Layered Composition. Context: This provides a visual continuity, showing the system being discussed is the one previously established as overwhelmingly complex.
Characters: Multiple Identical White Silhouettes - A series of minimalist, glowing white human silhouettes, identical in shape and posture. They are positioned in the foreground, performing a synchronized action. Style: A set of Minimalist 2D Vector Illustrations in the Kurzgesagt Style, designed as Motion Graphics Assets. Rendered with High Contrast and Glowing Light Effects on a Deep Black Canvas at 4K Resolution. Context: These silhouettes represent Hoenig's colleagues or "the consensus," all working together within the system.
Object: Large Central Gear - A single, large, minimalist gear, rendered in a solid, slightly glowing white or light grey. It is the central focal point for the silhouettes' action. Style: A Minimalist 2D Vector Illustration, Geometric Abstraction with Clean Lines. This is a Professional Grade, 4K Resolution Motion Graphics Asset in a Neo-Retro Futurism style. Context: This gear symbolizes a core mechanism of the financial system that the colleagues are manipulating.
Special Effect: Gear Crack - A jagged, dark crack that appears on the surface of the central gear. It should be designed as a separate layer to be animated appearing and widening. Style: A clean, sharp vector shape designed for a Layered Composition. This is a Motion Graphics Asset with High Contrast at 4K Resolution. Context: This visual element represents the beginning of a systemic failure.
Special Effect: Ominous Red Glow - A pulsing, Ominous Red Glow that emanates from within the gear crack. This should be a soft, light-emitting effect that can be animated. Style: A Glowing Light Effect asset, designed for a Layered Composition to be placed behind the crack layer. This is a 4K Resolution asset with a Smooth Render. Context: The red glow signifies the danger and profound mistake Hoenig believes is being made.

**Example JSON Response:**
{
  "prompts": [
    "Background: Complex Financial System Diagram...\\nCharacters: Multiple Identical White Silhouettes...\\n(this is a single string with newlines, containing the full example above)",
    "Object: A single, glowing red 'dissenting' gear...\\n(another complete multi-line prompt string for the next scene)"
  ]
}

**SCRIPT TO ANALYZE:**
---
${script}
---
`;

    try {
        const response = await ai.models.generateContent({
            model: PROMPT_GENERATION_MODEL,
            contents: requestPrompt,
            config: {
                responseMimeType: 'application/json',
                responseSchema: promptSchema,
            },
        });
        
        const jsonText = response.text;
        
        if (!jsonText) {
            console.error("AI response text is empty or undefined. Full response:", response);
            const finishReason = response.candidates?.[0]?.finishReason;
            let errorMessage = "The AI returned an empty response. This can happen if the script is too short, vague, or contains content that goes against the safety policy.";

            if (finishReason === 'SAFETY') {
                errorMessage = "The script or style contains content that violates safety policies. Please revise your input and try again.";
            } else if (finishReason === 'RECITATION') {
                errorMessage = "The response was blocked due to potential recitation issues. Try rephrasing your script.";
            } else if (finishReason && finishReason !== 'STOP') {
                errorMessage = `Prompt generation stopped unexpectedly. Reason: ${finishReason}. Please check your script content.`;
            }
            throw new Error(errorMessage);
        }

        try {
            const result = JSON.parse(jsonText);
            if (result && Array.isArray(result.prompts)) {
                return { prompts: result.prompts, requestPrompt };
            } else {
                throw new Error('The AI returned a response with an invalid structure. Please try again.');
            }
        } catch (parseError) {
             console.error("Failed to parse AI response as JSON:", jsonText, parseError);
             throw new Error("The AI returned a response that was not valid JSON. This may be a temporary issue, please try again.");
        }

    } catch (error) {
        console.error("Error during prompt generation:", error);
        // Re-throw specific, user-friendly errors, otherwise provide a generic one.
        if (error instanceof Error && (
            error.message.startsWith("The AI returned") ||
            error.message.startsWith("The script or style") ||
            error.message.startsWith("The response was blocked") ||
            error.message.startsWith("Prompt generation stopped")
        )) {
            throw error;
        }
        throw new Error("Failed to generate prompts from the script due to an unexpected AI service error.");
    }
}

export async function analyzeImageStyle(imageData: { base64: string, mimeType: string }): Promise<string> {
    const imagePart = {
        inlineData: {
            mimeType: imageData.mimeType,
            data: imageData.base64,
        },
    };
    const textPart = {
        text: "Analyze the artistic style of this image. Describe the style in a concise, comma-separated list of keywords and phrases suitable for a text-to-image AI. Focus on elements like lighting, color palette, composition, medium (e.g., photograph, oil painting), and overall mood. Do not use full sentences. Example: cinematic, dramatic lighting, high contrast, muted color palette, photorealistic, shallow depth of field, moody atmosphere.",
    };

    try {
        const response = await ai.models.generateContent({
            model: PROMPT_GENERATION_MODEL,
            contents: { parts: [imagePart, textPart] },
        });
        return response.text.trim();
    } catch (error) {
        console.error("Error analyzing image style:", error);
        throw new Error("Failed to analyze the reference image style.");
    }
}


export async function generateImage(prompt: string, aspectRatio: AspectRatio): Promise<{ base64: string | null; error: string | null; }> {
    try {
        const response = await ai.models.generateImages({
            model: IMAGE_GENERATION_MODEL,
            prompt: prompt,
            config: {
                numberOfImages: 1,
                outputMimeType: 'image/jpeg',
                aspectRatio: aspectRatio,
            }
        });

        if (response.generatedImages && response.generatedImages.length > 0) {
            return { base64: response.generatedImages[0].image.imageBytes, error: null };
        } else {
            return { base64: null, error: 'The API did not return an image.' };
        }
    } catch(err) {
        console.error(`Image generation failed for prompt: "${prompt}"`, err);
        const errorMessage = err instanceof Error ? err.message : 'An unknown error occurred.';
        if (errorMessage.includes("sensitive words") || errorMessage.includes("Responsible AI practices") || errorMessage.includes("prompt contains sensitive words")) {
             return { base64: null, error: 'This prompt was blocked for safety reasons. Please try rephrasing it.' };
        }
        return { base64: null, error: 'Image generation failed.' };
    }
}
