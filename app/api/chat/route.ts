import OpenAI from "openai";

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

export async function POST(req: Request) {
  const { messages } = await req.json();

  const response = await client.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [
      {
        role: "system",
        content: `
You are a professional driving exam assistant for Jordan.

You must answer in BOTH languages:
- First: Arabic
- Then: English

Keep answers short and clear.
        `,
      },
      ...messages
    ],
  });

  return Response.json({
    reply: response.choices[0].message.content,
  });
}