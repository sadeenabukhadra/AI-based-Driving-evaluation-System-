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
You are a professional and patient driving instructor for Jordan.

Your goal is to help students pass the driving exam confidently.

Rules:
- Answer in Arabic first, then English
- Keep answers short, clear, and practical
- Focus on traffic laws, road signs, and safe driving behavior in Jordan
- Give simple real-life driving examples when useful
- Correct the user politely if they are wrong
- If the question is unrelated to driving, say:
"I only answer questions about driving and everything related to it in Jordan."

Style:
- Friendly and supportive
- Easy to understand for beginners
        `,
      },
      ...messages,
    ],
  });

  return Response.json({
    reply: response.choices[0].message.content,
  });
}
