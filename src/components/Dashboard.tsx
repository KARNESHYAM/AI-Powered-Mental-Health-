import React, { useState, useEffect } from 'react';
import { collection, query, where, orderBy, limit, onSnapshot } from 'firebase/firestore';
import { db, handleFirestoreError, OperationType } from '../firebase';
import { useAuth } from '../App';
import { MoodEntry, JournalEntry } from '../types';
import { motion } from 'motion/react';
import { Smile, BookOpen, Wind, TrendingUp, Calendar } from 'lucide-react';
import { format } from 'date-fns';

interface DashboardProps {
  onNavigate: (tab: string) => void;
}

const Dashboard: React.FC<DashboardProps> = ({ onNavigate }) => {
  const { user, profile } = useAuth();
  const [recentMoods, setRecentMoods] = useState<MoodEntry[]>([]);
  const [recentJournals, setRecentJournals] = useState<JournalEntry[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!user) return;

    const moodsQuery = query(
      collection(db, 'moods'),
      where('uid', '==', user.uid),
      orderBy('timestamp', 'desc'),
      limit(5)
    );

    const journalsQuery = query(
      collection(db, 'journals'),
      where('uid', '==', user.uid),
      orderBy('timestamp', 'desc'),
      limit(3)
    );

    const unsubscribeMoods = onSnapshot(moodsQuery, (snapshot) => {
      setRecentMoods(snapshot.docs.map(doc => ({ id: doc.id, ...doc.data() } as MoodEntry)));
    }, (error) => {
      handleFirestoreError(error, OperationType.LIST, 'moods');
    });

    const unsubscribeJournals = onSnapshot(journalsQuery, (snapshot) => {
      setRecentJournals(snapshot.docs.map(doc => ({ id: doc.id, ...doc.data() } as JournalEntry)));
      setLoading(false);
    }, (error) => {
      handleFirestoreError(error, OperationType.LIST, 'journals');
    });

    return () => {
      unsubscribeMoods();
      unsubscribeJournals();
    };
  }, [user]);

  const getMoodEmoji = (mood: string) => {
    switch (mood) {
      case 'happy': return '😊';
      case 'calm': return '😌';
      case 'neutral': return '😐';
      case 'anxious': return '😰';
      case 'sad': return '😢';
      case 'angry': return '😠';
      default: return '😶';
    }
  };

  return (
    <div className="space-y-8">
      <header>
        <h1 className="text-3xl font-bold text-slate-800">Welcome back, {profile?.displayName?.split(' ')[0] || 'Friend'}</h1>
        <p className="text-slate-500">How are you feeling today?</p>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Quick Stats */}
        <motion.div 
          whileHover={{ y: -5 }}
          className="bg-white p-6 rounded-3xl shadow-sm border border-slate-100 flex items-center gap-4"
        >
          <div className="w-12 h-12 bg-emerald-100 rounded-2xl flex items-center justify-center text-emerald-600">
            <TrendingUp size={24} />
          </div>
          <div>
            <p className="text-sm text-slate-500">Mood Streak</p>
            <p className="text-2xl font-bold text-slate-800">5 Days</p>
          </div>
        </motion.div>

        <motion.div 
          whileHover={{ y: -5 }}
          className="bg-white p-6 rounded-3xl shadow-sm border border-slate-100 flex items-center gap-4"
        >
          <div className="w-12 h-12 bg-blue-100 rounded-2xl flex items-center justify-center text-blue-600">
            <BookOpen size={24} />
          </div>
          <div>
            <p className="text-sm text-slate-500">Journal Entries</p>
            <p className="text-2xl font-bold text-slate-800">{recentJournals.length}</p>
          </div>
        </motion.div>

        <motion.div 
          whileHover={{ y: -5 }}
          className="bg-white p-6 rounded-3xl shadow-sm border border-slate-100 flex items-center gap-4"
        >
          <div className="w-12 h-12 bg-purple-100 rounded-2xl flex items-center justify-center text-purple-600">
            <Wind size={24} />
          </div>
          <div>
            <p className="text-sm text-slate-500">Mindful Minutes</p>
            <p className="text-2xl font-bold text-slate-800">45</p>
          </div>
        </motion.div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Recent Moods */}
        <section className="bg-white p-8 rounded-3xl shadow-sm border border-slate-100">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-bold text-slate-800 flex items-center gap-2">
              <Smile className="text-emerald-500" /> Recent Moods
            </h2>
            <button 
              onClick={() => onNavigate('mood')}
              className="text-sm text-emerald-600 font-semibold hover:underline"
            >
              View All
            </button>
          </div>
          
          <div className="space-y-4">
            {recentMoods.length > 0 ? (
              recentMoods.map((entry) => (
                <div key={entry.id} className="flex items-center justify-between p-4 bg-slate-50 rounded-2xl">
                  <div className="flex items-center gap-4">
                    <span className="text-3xl">{getMoodEmoji(entry.mood)}</span>
                    <div>
                      <p className="font-semibold text-slate-800 capitalize">{entry.mood}</p>
                      <p className="text-xs text-slate-500">{entry.timestamp?.toDate ? format(entry.timestamp.toDate(), 'MMM d, h:mm a') : 'Just now'}</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <span className="text-sm font-medium text-slate-600">Intensity: {entry.intensity}/10</span>
                  </div>
                </div>
              ))
            ) : (
              <p className="text-center text-slate-400 py-8 italic">No mood entries yet. Log your first mood!</p>
            )}
          </div>
        </section>

        {/* Recent Journal */}
        <section className="bg-white p-8 rounded-3xl shadow-sm border border-slate-100">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-bold text-slate-800 flex items-center gap-2">
              <BookOpen className="text-blue-500" /> Recent Journals
            </h2>
            <button 
              onClick={() => onNavigate('journal')}
              className="text-sm text-blue-600 font-semibold hover:underline"
            >
              View All
            </button>
          </div>

          <div className="space-y-4">
            {recentJournals.length > 0 ? (
              recentJournals.map((entry) => (
                <div key={entry.id} className="p-4 bg-slate-50 rounded-2xl">
                  <h3 className="font-bold text-slate-800 mb-1">{entry.title || 'Untitled Entry'}</h3>
                  <p className="text-sm text-slate-600 line-clamp-2 mb-2">{entry.content}</p>
                  <div className="flex items-center justify-between">
                    <div className="flex gap-2">
                      {entry.tags?.slice(0, 2).map(tag => (
                        <span key={tag} className="text-[10px] px-2 py-0.5 bg-blue-100 text-blue-700 rounded-full font-medium uppercase tracking-wider">
                          {tag}
                        </span>
                      ))}
                    </div>
                    <span className="text-[10px] text-slate-400 uppercase font-bold">
                      {entry.timestamp?.toDate ? format(entry.timestamp.toDate(), 'MMM d') : 'Today'}
                    </span>
                  </div>
                </div>
              ))
            ) : (
              <p className="text-center text-slate-400 py-8 italic">Your journal is empty. Start writing your thoughts.</p>
            )}
          </div>
        </section>
      </div>

      {/* AI Suggestion Card */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-gradient-to-br from-emerald-500 to-teal-600 p-8 rounded-3xl text-white shadow-lg shadow-emerald-200"
      >
        <h2 className="text-2xl font-bold mb-2">Daily Reflection</h2>
        <p className="text-emerald-50 mb-6 max-w-2xl">
          "The best way to predict your future is to create it." Today seems like a great day to focus on small wins. How about a 5-minute breathing exercise to center yourself?
        </p>
        <button 
          onClick={() => onNavigate('exercises')}
          className="px-6 py-3 bg-white text-emerald-600 rounded-xl font-bold hover:bg-emerald-50 transition-colors"
        >
          Start Exercise
        </button>
      </motion.div>
    </div>
  );
};

export default Dashboard;
