import { GoogleGenAI } from "@google/genai";

const SYSTEM_PROMPT = `You are a compassionate AI mental health companion. 
You listen carefully, respond with empathy, and provide supportive suggestions using CBT techniques (Cognitive Behavioral Therapy). 
You never judge the user. 
You avoid giving medical diagnoses. 
You encourage healthy coping strategies and self-reflection.
Keep your responses concise but warm.`;

const API_KEY = import.meta.env.VITE_GEMINI_API_KEY;

const getGeminiTherapistResponse = async (history: { role: 'user' | 'model', content: string }[], message: string, userName?: string, image?: string) => {
  if (!API_KEY) {
    console.error("GEMINI_API_KEY is missing. Please set it in your environment variables (e.g., VITE_GEMINI_API_KEY for Netlify).");
    return "I'm sorry, I'm having trouble connecting to my AI brain right now (API key missing). But I'm still here to listen. How can I help you today?";
  }

  try {
    const ai = new GoogleGenAI({ apiKey: API_KEY });
    const contents: any[] = [];
    
    // Add history
    history.forEach(m => {
      contents.push({
        role: m.role,
        parts: [{ text: m.content }]
      });
    });
    
    // Add current message
    const currentParts: any[] = [];
    if (image) {
      const base64Data = image.split(',')[1];
      const mimeType = image.split(',')[0].split(':')[1].split(';')[0];
      currentParts.push({
        inlineData: {
          data: base64Data,
          mimeType: mimeType
        }
      });
    }
    currentParts.push({ text: message });
    
    contents.push({
      role: 'user',
      parts: currentParts
    });

    const result = await ai.models.generateContent({
      model: "gemini-3-flash-preview",
      contents: contents,
      config: {
        systemInstruction: `${SYSTEM_PROMPT} The user's name is ${userName || 'Friend'}. Address them by name when appropriate.`,
      }
    });

    return result.text || "I'm here for you. Could you tell me more about how you're feeling?";
  } catch (geminiError) {
    console.error("Gemini Fallback Error:", geminiError);
    return "I'm sorry, I'm having a bit of trouble connecting right now. But I'm still here for you. Please try again in a moment.";
  }
};

export const getTherapistResponse = async (history: { role: 'user' | 'model', content: string }[], message: string, userName?: string, image?: string) => {
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        messages: [...history, { role: 'user', content: message }],
        image,
        userName,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      if (errorData.error === "OPENAI_API_KEY_MISSING") {
        console.warn("OpenAI API key missing, using Gemini fallback.");
        return await getGeminiTherapistResponse(history, message, userName, image);
      }
      throw new Error(errorData.error || "Failed to get response from AI");
    }

    const data = await response.json();
    return data.response || "I'm here for you. Could you tell me more about how you're feeling?";
  } catch (error: any) {
    // If it's the specific OpenAI key error, we've already handled it or it's being thrown here
    if (error.message === "OPENAI_API_KEY_MISSING") {
      // This might happen if the error was thrown from line 62 but not caught by the if block
      return await getGeminiTherapistResponse(history, message, userName, image);
    }
    
    console.error("OpenAI Chat Error:", error);
    return await getGeminiTherapistResponse(history, message, userName, image);
  }
};

export const getSpeechFromText = async (text: string) => {
  try {
    const response = await fetch("/api/tts", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      throw new Error("Failed to get speech from AI");
    }

    const blob = await response.blob();
    return URL.createObjectURL(blob);
  } catch (error) {
    console.error("TTS Error:", error);
    return null;
  }
};

export const getJournalInsights = async (content: string) => {
  if (!API_KEY) return { insight: "Reflecting on your day is a great step towards self-awareness.", tags: ["reflection"] };

  try {
    const ai = new GoogleGenAI({ apiKey: API_KEY });
    const response = await ai.models.generateContent({
      model: "gemini-3-flash-preview",
      contents: [{ parts: [{ text: `Analyze this journal entry and provide a short, empathetic insight (2-3 sentences) and suggest 3 relevant tags. Format as JSON: { "insight": "...", "tags": ["tag1", "tag2", "tag3"] }\n\nEntry: ${content}` }] }],
      config: {
        responseMimeType: "application/json",
      },
    });

    return JSON.parse(response.text || "{}");
  } catch (error) {
    console.error("Journal Insight Error:", error);
    return { insight: "Reflecting on your day is a great step towards self-awareness.", tags: ["reflection"] };
  }
};
