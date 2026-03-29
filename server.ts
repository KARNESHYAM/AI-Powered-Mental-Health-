import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import OpenAI from "openai";
import dotenv from "dotenv";

dotenv.config();

let openaiClient: OpenAI | null = null;

function getOpenAI() {
  if (!openaiClient) {
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) {
      throw new Error("OPENAI_API_KEY_MISSING");
    }
    openaiClient = new OpenAI({ apiKey });
  }
  return openaiClient;
}

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json({ limit: '50mb' }));

  // API routes
  app.post("/api/chat", async (req, res) => {
    try {
      const openai = getOpenAI();
      const { messages, image, userName } = req.body;

      const userMessage: any = {
        role: "user",
        content: [
          { type: "text", text: messages[messages.length - 1].content },
        ],
      };

      if (image) {
        userMessage.content.push({
          type: "image_url",
          image_url: {
            url: image, // base64 data URL
          },
        });
      }

      const response = await openai.chat.completions.create({
        model: "gpt-4o",
        messages: [
          {
            role: "system",
            content: `You are a compassionate AI mental health companion. You listen carefully, respond with empathy, and provide supportive suggestions using CBT techniques (Cognitive Behavioral Therapy). You never judge the user. You avoid giving medical diagnoses. You encourage healthy coping strategies and self-reflection. Keep your responses concise but warm. The user's name is ${userName || 'Friend'}. Address them by name when appropriate.`,
          },
          ...messages.slice(0, -1).map((m: any) => ({
            role: m.role === 'model' ? 'assistant' : 'user',
            content: m.content,
          })),
          userMessage,
        ],
        max_tokens: 500,
      });

      const aiResponse = response.choices[0].message.content;
      res.json({ response: aiResponse });
    } catch (error: any) {
      if (error.message === "OPENAI_API_KEY_MISSING") {
        console.warn("OpenAI API key missing for chat request. Fallback will be handled by client.");
      } else {
        console.error("OpenAI Chat Error:", error);
      }
      res.status(500).json({ error: error.message });
    }
  });

  // TTS endpoint
  app.post("/api/tts", async (req, res) => {
    try {
      const openai = getOpenAI();
      const { text } = req.body;
      const mp3 = await openai.audio.speech.create({
        model: "tts-1",
        voice: "nova",
        input: text,
      });

      const buffer = Buffer.from(await mp3.arrayBuffer());
      res.set("Content-Type", "audio/mpeg");
      res.send(buffer);
    } catch (error: any) {
      if (error.message === "OPENAI_API_KEY_MISSING") {
        console.warn("OpenAI API key missing for TTS request");
      } else {
        console.error("OpenAI TTS Error:", error);
      }
      res.status(500).json({ error: error.message });
    }
  });

  // Vite middleware for development
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { 
        middlewareMode: true,
        hmr: false // Explicitly disable HMR to prevent websocket connection errors in AI Studio
      },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
