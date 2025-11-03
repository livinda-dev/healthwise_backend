import { defineFlow } from "@genkit-ai/flow";
import { googleAI } from "@genkit-ai/google-genai";
import { genkit } from 'genkit';
import { z } from "zod";

const ai = genkit({
  plugins: [
    googleAI({
      apiKey: process.env.GOOGLE_GENAI_API_KEY,
    }),
  ],
});


const messageSchema = z.object({
  messages: z.array(
    z.object({
      role: z.string(),
      parts: z.array(z.object({ text: z.string() })),
    })
  ),
});

export const healthAssistant = defineFlow(
  {
    name: "healthAssistant",
    inputSchema: messageSchema,
    outputSchema: z.object({
      text: z.string(),
    }),
  },
  async (input) => {
    const conversation = input.messages
      .map((m) => `${m.role}: ${m.parts.map((p) => p.text).join(" ")}`)
      .join("\n");

    const prompt = `
You are a helpful AI health companion. Provide friendly, informative responses about general health and wellness.
Always remind users to consult healthcare professionals for medical advice.
Keep responses concise and supportive.

Conversation so far:
${conversation}
`;

    const result = await ai.generate({
      model: 'models/gemini-1.5-flash',
      prompt,
      config: {
        temperature: 0.7,
      },
    });

    const text = result.text;

    return { text: text || "Sorry, I couldn’t generate a response." };
  }
);
