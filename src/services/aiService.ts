import { GoogleGenAI } from "@google/genai";
import firebaseConfig from '../../firebase-applet-config.json';

const SYSTEM_PROMPT = `You are a compassionate AI mental health companion. 
You listen carefully, respond with empathy, and provide supportive suggestions using CBT techniques (Cognitive Behavioral Therapy). 
You never judge the user. 
You avoid giving medical diagnoses. 
You encourage healthy coping strategies and self-reflection.
Keep your responses concise but warm.`;

// Use the VITE_ prefix for client-side environment variables
// In AI Studio Build, GEMINI_API_KEY is often injected directly.
const getApiKey = () => {
  // 1. Try VITE_ prefix (standard for client-side)
  const vKey = import.meta.env.VITE_GEMINI_API_KEY;
  if (vKey && vKey !== "undefined" && vKey !== "") return vKey;
  
  // 2. Try process.env (fallback for some environments)
  const pKey = typeof process !== 'undefined' ? process.env.GEMINI_API_KEY : undefined;
  if (pKey && pKey !== "undefined" && pKey !== "") return pKey;
  
  // 3. Try firebaseConfig.apiKey (user provided this in a previous request)
  if (firebaseConfig && firebaseConfig.apiKey && firebaseConfig.apiKey !== "undefined" && firebaseConfig.apiKey !== "") {
    return firebaseConfig.apiKey;
  }
  
  return undefined;
};

const API_KEY = getApiKey();

const getGeminiTherapistResponse = async (
  history: { role: 'user' | 'model', content: string }[], 
  message: string, 
  userName?: string, 
  image?: string,
  onChunk?: (chunk: string) => void
) => {
  if (!API_KEY) {
    console.error("Gemini API Key is missing. Please check your environment variables (VITE_GEMINI_API_KEY or GEMINI_API_KEY).");
    return "I'm sorry, I'm having trouble connecting to my AI brain right now. The API key seems to be missing. Please check your environment settings.";
  }

  const contents: any[] = [];
  try {
    const ai = new GoogleGenAI({ apiKey: API_KEY });
    
    // Filter and format history to ensure alternating roles and that it starts with a user message
    let lastRole: string | null = null;
    history.forEach(m => {
      if (!m.content?.trim()) return; // Skip empty messages
      
      // Map 'model' to 'model' (Gemini SDK uses 'model' for the assistant role)
      const role = m.role === 'model' ? 'model' : 'user';
      
      // Gemini requires the first message to be from the 'user'
      if (contents.length === 0 && role === 'model') {
        return; // Skip if the first message in history is from the model
      }
      
      if (role === lastRole) {
        // Merge consecutive messages from the same role
        if (contents.length > 0) {
          const lastContent = contents[contents.length - 1];
          const lastPart = lastContent.parts[lastContent.parts.length - 1];
          if (lastPart.text) {
            lastPart.text += "\n\n" + m.content;
          } else {
            lastContent.parts.push({ text: m.content });
          }
        }
        return;
      }
      
      contents.push({
        role: role,
        parts: [{ text: m.content }]
      });
      lastRole = role;
    });
    
    // Add current message
    const currentParts: any[] = [];
    if (image) {
      try {
        // Robust base64 extraction
        const base64Data = image.includes(',') ? image.split(',')[1] : image;
        const mimeTypeMatch = image.match(/^data:([^;]+);/);
        const mimeType = mimeTypeMatch ? mimeTypeMatch[1] : "image/png";
        
        currentParts.push({
          inlineData: {
            data: base64Data,
            mimeType: mimeType
          }
        });
      } catch (e) {
        console.error("Image processing error:", e);
      }
    }
    
    if (message?.trim()) {
      currentParts.push({ text: message });
    } else if (currentParts.length === 0) {
      currentParts.push({ text: "Please look at this image." });
    }
    
    if (lastRole === 'user') {
      // If the last message was from user, merge the current message into it
      if (contents.length > 0) {
        contents[contents.length - 1].parts.push(...currentParts);
      } else {
        contents.push({ role: 'user', parts: currentParts });
      }
    } else {
      contents.push({
        role: 'user',
        parts: currentParts
      });
    }

    // Using gemini-3-flash-preview as it's the most stable for AI Studio Build
    const modelName = "gemini-3-flash-preview"; 

    console.log("Attempting Gemini request:", {
      model: modelName,
      historyCount: contents.length,
      hasImage: !!image,
      hasOnChunk: !!onChunk
    });

    // Use streaming for fast responses if a callback is provided
    if (onChunk) {
      const result = await ai.models.generateContentStream({
        model: modelName,
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
        model: modelName,
        contents: contents,
        config: {
          systemInstruction: `${SYSTEM_PROMPT} The user's name is ${userName || 'Friend'}. Address them by name when appropriate.`,
        }
      });

      return result.text || "I'm here for you. Could you tell me more about how you're feeling?";
    }
  } catch (geminiError: any) {
    console.error("Gemini API Error:", {
      name: geminiError?.name,
      message: geminiError?.message,
      status: geminiError?.status,
      details: geminiError?.details,
      stack: geminiError?.stack
    });
    
    // If it's a model not found error, try one more fallback
    if (geminiError?.message?.includes('not found') || geminiError?.message?.includes('404')) {
      console.warn("Model gemini-3-flash-preview not found, trying gemini-1.5-flash as fallback...");
      try {
        const ai = new GoogleGenAI({ apiKey: API_KEY });
        const modelName = "gemini-1.5-flash";
        if (onChunk) {
          const result = await ai.models.generateContentStream({
            model: modelName,
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
            model: modelName,
            contents: contents,
            config: {
              systemInstruction: `${SYSTEM_PROMPT} The user's name is ${userName || 'Friend'}. Address them by name when appropriate.`,
            }
          });
          return result.text || "I'm here for you.";
        }
      } catch (fallbackError) {
        console.error("Gemini Fallback Error:", fallbackError);
      }
    }
    
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
  // ALWAYS prioritize Gemini if onChunk is provided for a dynamic experience
  // OR if OpenAI is known to be broken
  if (onChunk || skipOpenAI) {
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
