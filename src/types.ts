export type MoodType = 'happy' | 'calm' | 'neutral' | 'anxious' | 'sad' | 'angry';

export interface UserProfile {
  uid: string;
  email: string;
  displayName?: string;
  photoURL?: string;
  phoneNumber?: string;
  createdAt: string;
  lastLogin?: string;
  role?: string;
  settings: {
    darkMode: boolean;
    notifications: boolean;
  };
}

export interface MoodEntry {
  id?: string;
  uid: string;
  mood: MoodType;
  intensity: number;
  note: string;
  timestamp: any; // Firestore Timestamp
}

export interface JournalEntry {
  id?: string;
  uid: string;
  title: string;
  content: string;
  mood?: string;
  tags: string[];
  aiInsights?: string;
  timestamp: any; // Firestore Timestamp
}

export interface ChatMessage {
  id?: string;
  role: 'user' | 'model';
  content: string;
  image?: string;
  timestamp: any; // Firestore Timestamp
}

export interface ChatSession {
  id?: string;
  uid: string;
  title: string;
  lastMessage: string;
  updatedAt: any; // Firestore Timestamp
  createdAt: any; // Firestore Timestamp
}
