import React, { useState, useEffect, useRef } from 'react';
import { collection, query, orderBy, onSnapshot, addDoc, serverTimestamp, doc, updateDoc, where, getDocs, writeBatch } from 'firebase/firestore';
import { db, handleFirestoreError, OperationType } from '../firebase';
import { useAuth } from '../App';
import { ChatMessage, ChatSession } from '../types';
import { getTherapistResponse, getSpeechFromText } from '../services/aiService';
import { Send, User, Bot, Loader2, Plus, Image as ImageIcon, Mic, MicOff, Volume2, X, Wind, Heart, Trash2 } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import ReactMarkdown from 'react-markdown';
import { format } from 'date-fns';
import ConfirmModal from './ConfirmModal';

const Chat: React.FC = () => {
  const { user, profile } = useAuth();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState<string | null>(null);
  const [micError, setMicError] = useState<string | null>(null);
  const [sessionToDelete, setSessionToDelete] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const recognitionRef = useRef<any>(null);

  useEffect(() => {
    if (!user) return;

    const sessionsQuery = query(
      collection(db, 'chatSessions'),
      where('uid', '==', user.uid),
      orderBy('updatedAt', 'desc')
    );

    const unsubscribe = onSnapshot(sessionsQuery, (snapshot) => {
      const sessionList = snapshot.docs.map(doc => ({ id: doc.id, ...doc.data() } as ChatSession));
      setSessions(sessionList);
      if (sessionList.length > 0 && !activeSessionId) {
        setActiveSessionId(sessionList[0].id!);
      }
    }, (error) => {
      handleFirestoreError(error, OperationType.LIST, 'chatSessions');
    });

    return unsubscribe;
  }, [user]);

  useEffect(() => {
    if (!activeSessionId) return;

    const messagesQuery = query(
      collection(db, 'chatSessions', activeSessionId, 'messages'),
      orderBy('timestamp', 'asc')
    );

    const unsubscribe = onSnapshot(messagesQuery, (snapshot) => {
      setMessages(snapshot.docs.map(doc => ({ id: doc.id, ...doc.data() } as ChatMessage)));
    }, (error) => {
      handleFirestoreError(error, OperationType.LIST, `chatSessions/${activeSessionId}/messages`);
    });

    return unsubscribe;
  }, [activeSessionId]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Speech Recognition Setup
  useEffect(() => {
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (SpeechRecognition) {
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = false;
      recognitionRef.current.interimResults = false;
      recognitionRef.current.lang = 'en-US';

      recognitionRef.current.maxAlternatives = 1;

      recognitionRef.current.onstart = () => {
        setIsRecording(true);
      };

      recognitionRef.current.onresult = (event: any) => {
        const transcript = event.results[0][0].transcript;
        setInput(prev => prev + ' ' + transcript);
      };

      recognitionRef.current.onerror = (event: any) => {
        console.error('Speech recognition error:', event.error);
        if (event.error === 'not-allowed') {
          setMicError('Microphone access was denied. Please check your browser permissions.');
        } else if (event.error === 'network') {
          setMicError('Network error: The speech recognition service is unreachable. Please check your internet connection or try again.');
        } else if (event.error === 'no-speech') {
          // No speech detected, just stop silently
          return;
        } else {
          setMicError(`Speech recognition error: ${event.error}`);
        }
        setTimeout(() => setMicError(null), 5000);
      };

      recognitionRef.current.onend = () => {
        setIsRecording(false);
      };
    } else {
      console.warn('Speech recognition not supported in this browser.');
    }

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.onstart = null;
        recognitionRef.current.onresult = null;
        recognitionRef.current.onerror = null;
        recognitionRef.current.onend = null;
        try {
          recognitionRef.current.abort();
        } catch (e) {
          // Ignore errors on abort
        }
      }
    };
  }, []);

  const toggleRecording = () => {
    if (!recognitionRef.current) {
      setMicError('Speech recognition is not supported in your browser. Please try Chrome or Edge.');
      setTimeout(() => setMicError(null), 5000);
      return;
    }

    if (isRecording) {
      recognitionRef.current?.stop();
    } else {
      try {
        recognitionRef.current?.start();
      } catch (err) {
        console.error('Failed to start speech recognition:', err);
        setIsRecording(false);
      }
    }
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setSelectedImage(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const playVoice = async (text: string, messageId: string) => {
    if (isSpeaking === messageId) {
      setIsSpeaking(null);
      window.speechSynthesis.cancel();
      return;
    }

    setIsSpeaking(messageId);
    const audioUrl = await getSpeechFromText(text);
    if (audioUrl) {
      const audio = new Audio(audioUrl);
      audio.onended = () => setIsSpeaking(null);
      audio.play();
    } else {
      // Fallback to browser TTS
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.onend = () => setIsSpeaking(null);
      window.speechSynthesis.speak(utterance);
    }
  };

  const createNewSession = async () => {
    if (!user) return;
    const newSession = {
      uid: user.uid,
      title: 'New Conversation',
      lastMessage: '',
      createdAt: serverTimestamp(),
      updatedAt: serverTimestamp()
    };
    try {
      const docRef = await addDoc(collection(db, 'chatSessions'), newSession);
      setActiveSessionId(docRef.id);
      setMessages([]);
    } catch (error) {
      handleFirestoreError(error, OperationType.CREATE, 'chatSessions');
    }
  };

  const deleteSession = async (id: string) => {
    try {
      // Delete messages subcollection
      const messagesSnap = await getDocs(collection(db, 'chatSessions', id, 'messages'));
      const batch = writeBatch(db);
      messagesSnap.docs.forEach(m => batch.delete(m.ref));
      // Delete session
      batch.delete(doc(db, 'chatSessions', id));
      await batch.commit();
      
      if (activeSessionId === id) {
        setActiveSessionId(null);
        setMessages([]);
      }
      setSessionToDelete(null);
    } catch (error) {
      handleFirestoreError(error, OperationType.DELETE, `chatSessions/${id}`);
    }
  };

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    if ((!input.trim() && !selectedImage) || !user) return;

    let sessionId = activeSessionId;
    
    // Auto-create session if none exists
    if (!sessionId) {
      const newSession = {
        uid: user.uid,
        title: (input.trim().slice(0, 30) || 'New Conversation') + '...',
        lastMessage: '',
        createdAt: serverTimestamp(),
        updatedAt: serverTimestamp()
      };
      try {
        const docRef = await addDoc(collection(db, 'chatSessions'), newSession);
        sessionId = docRef.id;
        setActiveSessionId(sessionId);
      } catch (error) {
        handleFirestoreError(error, OperationType.CREATE, 'chatSessions');
        return;
      }
    }

    const userMessage = input.trim();
    const currentImage = selectedImage;
    setInput('');
    setSelectedImage(null);
    setIsTyping(true);

    try {
      // Save user message
      await addDoc(collection(db, 'chatSessions', sessionId, 'messages'), {
        role: 'user',
        content: userMessage,
        image: currentImage,
        timestamp: serverTimestamp()
      });

      // Update session last message
      const updateData: any = {
        lastMessage: userMessage || 'Sent an image',
        updatedAt: serverTimestamp(),
      };
      
      // Only update title if it's the first message
      if (messages.length === 0) {
        updateData.title = (userMessage.slice(0, 30) || 'Image Session') + '...';
      }

      await updateDoc(doc(db, 'chatSessions', sessionId), updateData);

      // Get AI response
      const history = messages.map(m => ({ role: m.role, content: m.content }));
      const aiResponse = await getTherapistResponse(history, userMessage, profile?.displayName || undefined, currentImage || undefined);

      // Save AI message
      await addDoc(collection(db, 'chatSessions', sessionId, 'messages'), {
        role: 'model',
        content: aiResponse,
        timestamp: serverTimestamp()
      });

    } catch (error) {
      console.error("Chat error:", error);
      if (error instanceof Error && error.message.includes('Missing or insufficient permissions')) {
        handleFirestoreError(error, OperationType.WRITE, `chatSessions/${sessionId}`);
      }
    } finally {
      setIsTyping(false);
    }
  };

  return (
    <div className="relative flex h-[calc(100vh-120px)] bg-slate-50 rounded-[2.5rem] shadow-2xl border border-white/20 overflow-hidden backdrop-blur-sm">
      {/* Immersive Background Elements */}
      <div className="absolute inset-0 -z-10 overflow-hidden pointer-events-none">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-emerald-200/30 rounded-full blur-[120px]" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-blue-200/30 rounded-full blur-[120px]" />
      </div>

      {/* Sessions Sidebar */}
      <div className="w-72 border-r border-slate-200/50 flex flex-col hidden md:flex bg-white/40 backdrop-blur-md">
        <div className="p-6 border-b border-slate-200/50">
          <button 
            onClick={createNewSession}
            className="w-full py-3 px-4 bg-emerald-600 text-white rounded-2xl font-bold flex items-center justify-center gap-2 hover:bg-emerald-700 transition-all shadow-lg shadow-emerald-100"
          >
            <Plus size={18} /> New Session
          </button>
        </div>
        <div className="flex-1 overflow-y-auto p-4 space-y-2">
          <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest px-2 mb-2">Recent Conversations</h3>
          {sessions.map(session => (
            <div
              key={session.id}
              onClick={() => setActiveSessionId(session.id!)}
              className={`w-full text-left p-4 rounded-2xl transition-all group cursor-pointer relative ${
                activeSessionId === session.id 
                  ? 'bg-white text-slate-900 shadow-md ring-1 ring-slate-200/50 font-bold' 
                  : 'text-slate-500 hover:bg-white/60'
              }`}
            >
              <div className="pr-8">
                <p className="text-sm truncate mb-1">{session.title}</p>
                <p className="text-[10px] text-slate-400 truncate group-hover:text-slate-500 transition-colors">
                  {session.lastMessage || 'Start a new conversation'}
                </p>
              </div>
              <button 
                onClick={(e) => {
                  e.stopPropagation();
                  setSessionToDelete(session.id!);
                }}
                className="absolute right-3 top-1/2 -translate-y-1/2 p-2 text-slate-300 hover:text-rose-500 opacity-0 group-hover:opacity-100 transition-all"
              >
                <Trash2 size={14} />
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* Chat Area */}
      <div className="flex-1 flex flex-col min-w-0 bg-transparent">
        {/* Chat Header */}
        <header className="p-6 border-b border-slate-200/50 bg-white/40 backdrop-blur-md flex items-center justify-between relative overflow-hidden">
          {/* Subtle Breathing Background */}
          <motion.div 
            animate={{ scale: [1, 1.05, 1], opacity: [0.1, 0.2, 0.1] }}
            transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }}
            className="absolute inset-0 bg-emerald-100/50 -z-10"
          />
          
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 bg-emerald-100 rounded-2xl flex items-center justify-center text-emerald-600 shadow-inner relative">
              <Bot size={24} fill="currentColor" />
              <div className="absolute -bottom-1 -right-1 w-4 h-4 bg-emerald-500 border-2 border-white rounded-full" />
            </div>
            <div>
              <h2 className="font-bold text-slate-800">AI Therapist</h2>
              <div className="flex items-center gap-1.5">
                <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />
                <span className="text-[10px] font-bold text-emerald-600 uppercase tracking-wider">Online & Listening</span>
              </div>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            <div className="hidden sm:flex flex-col items-end">
              <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Safe Space</span>
              <span className="text-[9px] text-slate-300">End-to-end Encrypted</span>
            </div>
            <div className="w-8 h-8 rounded-full bg-slate-100 flex items-center justify-center text-slate-400">
              <Heart size={14} fill="currentColor" />
            </div>
          </div>
        </header>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6 scrollbar-hide">
          {messages.length === 0 && !isTyping ? (
            <div className="h-full flex flex-col items-center justify-center text-center max-w-sm mx-auto space-y-6">
              <div className="w-24 h-24 bg-white/60 rounded-[2rem] flex items-center justify-center text-emerald-500 shadow-xl border border-white">
                <Wind size={48} className="animate-pulse" />
              </div>
              <div>
                <h3 className="text-xl font-bold text-slate-800 mb-2">How can I help you today?</h3>
                <p className="text-sm text-slate-500 leading-relaxed">
                  I'm here to listen and support you. You can talk about your feelings, share your day, or ask for guidance.
                </p>
              </div>
              <div className="grid grid-cols-2 gap-3 w-full">
                {['I feel anxious', 'Help me sleep', 'Need to talk', 'Daily check-in'].map(suggestion => (
                  <button 
                    key={suggestion}
                    onClick={() => setInput(suggestion)}
                    className="p-3 bg-white/60 hover:bg-white text-xs font-bold text-slate-600 rounded-xl border border-white shadow-sm transition-all"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((msg, idx) => (
              <motion.div 
                key={msg.id || idx}
                initial={{ opacity: 0, y: 10, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div className={`max-w-[85%] md:max-w-[70%] space-y-2 group relative`}>
                  <div className={`p-5 rounded-[2rem] shadow-sm transition-all hover:shadow-md ${
                    msg.role === 'user' 
                      ? 'bg-emerald-600 text-white rounded-tr-none' 
                      : 'bg-white text-slate-800 rounded-tl-none border border-slate-100'
                  }`}>
                    {msg.image && (
                      <img src={msg.image} className="max-w-full rounded-2xl mb-3 border border-white/20 shadow-lg" alt="User upload" referrerPolicy="no-referrer" />
                    )}
                    <div className={`prose prose-sm max-w-none prose-p:leading-relaxed prose-p:m-0 ${msg.role === 'user' ? 'prose-invert' : 'prose-slate'}`}>
                      <ReactMarkdown>{msg.content}</ReactMarkdown>
                    </div>
                  </div>
                  <div className={`flex items-center gap-2 px-2 ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
                    <p className={`text-[10px] font-bold text-slate-400 uppercase tracking-tighter`}>
                      {msg.role === 'user' ? 'You' : 'Companion'}
                    </p>
                    {msg.role === 'model' && (
                      <button 
                        onClick={() => playVoice(msg.content, msg.id!)}
                        className={`p-1.5 rounded-full transition-all ${isSpeaking === msg.id ? 'bg-emerald-100 text-emerald-600' : 'text-slate-400 hover:bg-slate-100 opacity-0 group-hover:opacity-100'}`}
                      >
                        <Volume2 size={12} className={isSpeaking === msg.id ? 'animate-pulse' : ''} />
                      </button>
                    )}
                    <span className="text-[9px] text-slate-300">
                      {msg.timestamp?.toDate ? format(msg.timestamp.toDate(), 'h:mm a') : 'Just now'}
                    </span>
                  </div>
                </div>
              </motion.div>
            ))
          )}
          {isTyping && (
            <div className="flex justify-start">
              <div className="bg-white p-4 rounded-[2rem] rounded-tl-none border border-slate-100 shadow-sm flex gap-1.5">
                <span className="w-1.5 h-1.5 bg-slate-300 rounded-full animate-bounce" />
                <span className="w-1.5 h-1.5 bg-slate-300 rounded-full animate-bounce [animation-delay:0.2s]" />
                <span className="w-1.5 h-1.5 bg-slate-300 rounded-full animate-bounce [animation-delay:0.4s]" />
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="p-6 bg-white/40 backdrop-blur-md border-t border-slate-200/50 space-y-4">
          <AnimatePresence>
            {selectedImage && (
              <motion.div 
                initial={{ opacity: 0, scale: 0.9, y: 10 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.9, y: 10 }}
                className="relative inline-block"
              >
                <img src={selectedImage} className="w-24 h-24 object-cover rounded-2xl border-2 border-emerald-500 shadow-xl" alt="Preview" referrerPolicy="no-referrer" />
                <button 
                  onClick={() => setSelectedImage(null)}
                  className="absolute -top-2 -right-2 bg-rose-500 text-white rounded-full p-1.5 shadow-lg hover:bg-rose-600 transition-colors"
                >
                  <X size={14} />
                </button>
              </motion.div>
            )}
          </AnimatePresence>

          <form 
            onSubmit={handleSend}
            className="relative flex items-end gap-3 max-w-4xl mx-auto"
          >
            <input
              type="file"
              accept="image/*"
              className="hidden"
              ref={fileInputRef}
              onChange={handleImageUpload}
            />
            
            <div className="flex gap-1">
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="p-4 text-slate-400 hover:text-emerald-600 hover:bg-emerald-50 rounded-2xl transition-all"
              >
                <ImageIcon size={20} />
              </button>
              <div className="relative">
                <button
                  type="button"
                  onClick={toggleRecording}
                  className={`p-4 transition-all rounded-2xl ${isRecording ? 'bg-rose-100 text-rose-600 animate-pulse' : 'text-slate-400 hover:text-emerald-600 hover:bg-emerald-50'}`}
                >
                  {isRecording ? <MicOff size={20} /> : <Mic size={20} />}
                </button>
                <AnimatePresence>
                  {micError && (
                    <motion.div 
                      initial={{ opacity: 0, y: 10, scale: 0.9 }}
                      animate={{ opacity: 1, y: 0, scale: 1 }}
                      exit={{ opacity: 0, y: 10, scale: 0.9 }}
                      className="absolute bottom-full mb-4 left-0 w-64 bg-rose-500 text-white text-[10px] p-3 rounded-xl shadow-2xl z-50 font-bold"
                    >
                      {micError}
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </div>

            <div className="flex-1 relative group">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSend(e as any);
                  }
                }}
                placeholder="Share your thoughts..."
                className="w-full p-4 bg-white rounded-2xl border border-slate-200 focus:ring-2 focus:ring-emerald-500 focus:border-transparent outline-none resize-none min-h-[56px] max-h-32 transition-all shadow-sm group-hover:shadow-md"
                rows={1}
              />
            </div>
            <button
              type="submit"
              disabled={(!input.trim() && !selectedImage) || isTyping}
              className="p-4 bg-emerald-600 text-white rounded-2xl font-bold shadow-lg shadow-emerald-100 hover:bg-emerald-700 disabled:opacity-50 disabled:shadow-none transition-all"
            >
              <Send size={20} />
            </button>
          </form>
          <p className="text-[10px] text-center text-slate-400 mt-4 font-medium">
            Your conversations are private and encrypted.
          </p>
        </div>
      </div>
      <ConfirmModal 
        isOpen={!!sessionToDelete}
        title="Delete Conversation?"
        message="This will permanently remove this conversation and all its messages. This action cannot be undone."
        confirmLabel="Delete"
        onConfirm={() => sessionToDelete && deleteSession(sessionToDelete)}
        onCancel={() => setSessionToDelete(null)}
      />
    </div>
  );
};

export default Chat;
