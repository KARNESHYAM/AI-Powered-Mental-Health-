import { GoogleGenAI } from "@google/genai";
import firebaseConfig from '../../firebase-applet-config.json';

const SYSTEM_PROMPT = `You are a compassionate AI mental health companion. 
You listen carefully, respond with empathy, and provide supportive suggestions using CBT techniques (Cognitive Behavioral Therapy). 
You never judge the user. 
You avoid giving medical diagnoses. 
You encourage healthy coping strategies and self-reflection.
Keep your responses concise but warm.`;

const API_KEY = import.meta.env.VITE_GEMINI_API_KEY || firebaseConfig.apiKey;

const getGeminiTherapistResponse = async (
  history: { role: 'user' | 'model', content: string }[], 
  message: string, 
  userName?: string, 
  image?: string,
  onChunk?: (chunk: string) => void
) => {
  if (!API_KEY) {
    console.error("GEMINI_API_KEY is missing. Please set it in your environment variables.");
    return "I'm sorry, I'm having trouble connecting to my AI brain right now. But I'm still here to listen. How can I help you today?";
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

    // Use streaming for fast responses if a callback is provided
    if (onChunk) {
      const result = await ai.models.generateContentStream({
        model: "gemini-3-flash-preview",
        contents: contents,
        config: {
          systemInstruction: `${SYSTEM_PROMPT} The user's name is ${userName || 'Friend'}. Address them by name when appropriate.`,
        }
      });

      let fullText = "";
      for await (const chunk of result) {
        const chunkText = chunk.text || "";
        fullText += chunkText;
        onChunk(chunkText);
      }
      return fullText;
    } else {
      const result = await ai.models.generateContent({
        model: "gemini-3-flash-preview",
        contents: contents,
        config: {
          systemInstruction: `${SYSTEM_PROMPT} The user's name is ${userName || 'Friend'}. Address them by name when appropriate.`,
        }
      });

      return result.text || "I'm here for you. Could you tell me more about how you're feeling?";
    }
  } catch (geminiError) {
    console.error("Gemini Error:", geminiError);
    return "I'm sorry, I'm having a bit of trouble connecting right now. But I'm still here for you. Please try again in a moment.";
  }
};

let skipOpenAI = false;

export const getTherapistResponse = async (
  history: { role: 'user' | 'model', content: string }[], 
  message: string, 
  userName?: string, 
  image?: string,
  onChunk?: (chunk: string) => void
) => {
  // If OpenAI is known to be broken, go straight to Gemini
  if (skipOpenAI) {
    return await getGeminiTherapistResponse(history, message, userName, image, onChunk);
  }

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
      const isAuthError = errorData.error === "OPENAI_API_KEY_INVALID" || errorData.error === "OPENAI_AUTH_ERROR";
      
      if (isAuthError) {
        console.warn("OpenAI API key issue, falling back to Gemini and skipping OpenAI for future requests.");
        skipOpenAI = true; // Remember to skip OpenAI for the rest of the session
        return await getGeminiTherapistResponse(history, message, userName, image, onChunk);
      }
      throw new Error(errorData.error || "Failed to get response from AI");
    }

    const data = await response.json();
    const fullResponse = data.response || "I'm here for you. Could you tell me more about how you're feeling?";
    
    // If a chunk callback is provided, simulate a single chunk for compatibility
    if (onChunk) onChunk(fullResponse);
    
    return fullResponse;
  } catch (error: any) {
    console.error("OpenAI Chat Error:", error);
    // If any error occurs, fallback to Gemini
    return await getGeminiTherapistResponse(history, message, userName, image, onChunk);
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
