import React, { useState, useEffect } from 'react';
import { collection, addDoc, serverTimestamp, query, where, orderBy, onSnapshot } from 'firebase/firestore';
import { db, handleFirestoreError, OperationType } from '../firebase';
import { useAuth } from '../App';
import { MoodEntry, MoodType } from '../types';
import { motion } from 'motion/react';
import { Smile, Frown, Meh, CloudRain, Zap, Heart, History, TrendingUp } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { format } from 'date-fns';

const MoodTracker: React.FC = () => {
  const { user } = useAuth();
  const [mood, setMood] = useState<MoodType>('neutral');
  const [intensity, setIntensity] = useState(5);
  const [note, setNote] = useState('');
  const [history, setHistory] = useState<MoodEntry[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    if (!user) return;

    const q = query(
      collection(db, 'moods'),
      where('uid', '==', user.uid),
      orderBy('timestamp', 'asc')
    );

    const unsubscribe = onSnapshot(q, (snapshot) => {
      setHistory(snapshot.docs.map(doc => ({ id: doc.id, ...doc.data() } as MoodEntry)));
    }, (error) => {
      handleFirestoreError(error, OperationType.LIST, 'moods');
    });

    return unsubscribe;
  }, [user]);

  const moods: { type: MoodType; icon: any; color: string; label: string }[] = [
    { type: 'happy', icon: Smile, color: 'text-yellow-500 bg-yellow-50', label: 'Happy' },
    { type: 'calm', icon: Heart, color: 'text-emerald-500 bg-emerald-50', label: 'Calm' },
    { type: 'neutral', icon: Meh, color: 'text-slate-500 bg-slate-50', label: 'Neutral' },
    { type: 'anxious', icon: Zap, color: 'text-orange-500 bg-orange-50', label: 'Anxious' },
    { type: 'sad', icon: CloudRain, color: 'text-blue-500 bg-blue-50', label: 'Sad' },
    { type: 'angry', icon: Zap, color: 'text-rose-500 bg-rose-50', label: 'Angry' },
  ];

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!user) return;
    setIsSubmitting(true);

    try {
      await addDoc(collection(db, 'moods'), {
        uid: user.uid,
        mood,
        intensity,
        note,
        timestamp: serverTimestamp()
      });
      setNote('');
      setIntensity(5);
    } catch (error) {
      handleFirestoreError(error, OperationType.CREATE, 'moods');
    } finally {
      setIsSubmitting(false);
    }
  };

  const chartData = history.map(entry => ({
    date: entry.timestamp?.toDate ? format(entry.timestamp.toDate(), 'MM/dd') : '',
    intensity: entry.intensity,
    mood: entry.mood
  }));

  return (
    <div className="space-y-8">
      <header>
        <h1 className="text-3xl font-bold text-slate-800 dark:text-slate-100">Mood Tracker</h1>
        <p className="text-slate-500 dark:text-slate-400">Track your emotional journey and find patterns.</p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Log Mood */}
        <section className="lg:col-span-1 bg-white dark:bg-slate-800 p-8 rounded-3xl shadow-sm border border-slate-100 dark:border-slate-700">
          <h2 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-6 flex items-center gap-2">
            <Smile className="text-emerald-500" /> Log Your Mood
          </h2>
          
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="grid grid-cols-3 gap-3">
              {moods.map((m) => (
                <button
                  key={m.type}
                  type="button"
                  onClick={() => setMood(m.type)}
                  className={`flex flex-col items-center gap-2 p-3 rounded-2xl border-2 transition-all ${
                    mood === m.type 
                      ? 'border-emerald-500 bg-emerald-50 dark:bg-emerald-900/20' 
                      : 'border-transparent bg-slate-50 dark:bg-slate-900/50 hover:bg-slate-100 dark:hover:bg-slate-900'
                  }`}
                >
                  <m.icon className={m.color.split(' ')[0]} size={24} />
                  <span className="text-[10px] font-bold uppercase tracking-wider text-slate-600 dark:text-slate-400">{m.label}</span>
                </button>
              ))}
            </div>

            <div className="space-y-3">
              <div className="flex justify-between text-sm font-semibold text-slate-700 dark:text-slate-300">
                <label>Intensity</label>
                <span>{intensity}/10</span>
              </div>
              <input 
                type="range" 
                min="1" 
                max="10" 
                value={intensity}
                onChange={(e) => setIntensity(parseInt(e.target.value))}
                className="w-full h-2 bg-slate-100 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-emerald-600"
              />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-semibold text-slate-700 dark:text-slate-300">Notes (Optional)</label>
              <textarea
                value={note}
                onChange={(e) => setNote(e.target.value)}
                placeholder="What's making you feel this way?"
                className="w-full bg-slate-50 dark:bg-slate-900/50 border-none rounded-2xl p-4 text-sm focus:ring-2 focus:ring-emerald-500 min-h-[100px] dark:text-slate-200 dark:placeholder:text-slate-600"
              />
            </div>

            <button
              type="submit"
              disabled={isSubmitting}
              className="w-full py-4 bg-emerald-600 text-white rounded-2xl font-bold hover:bg-emerald-700 transition-all shadow-lg shadow-emerald-100 dark:shadow-none disabled:opacity-50"
            >
              {isSubmitting ? 'Logging...' : 'Log Mood'}
            </button>
          </form>
        </section>

        {/* Analytics */}
        <section className="lg:col-span-2 bg-white dark:bg-slate-800 p-8 rounded-3xl shadow-sm border border-slate-100 dark:border-slate-700">
          <h2 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-6 flex items-center gap-2">
            <TrendingUp className="text-blue-500" /> Mood Trends
          </h2>

          <div className="h-[300px] w-full">
            {history.length > 1 ? (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={chartData}>
                  <defs>
                    <linearGradient id="colorIntensity" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#10b981" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" className="dark:stroke-slate-700" />
                  <XAxis dataKey="date" axisLine={false} tickLine={false} tick={{fontSize: 12, fill: '#94a3b8'}} />
                  <YAxis axisLine={false} tickLine={false} tick={{fontSize: 12, fill: '#94a3b8'}} domain={[0, 10]} />
                  <Tooltip 
                    contentStyle={{ borderRadius: '16px', border: 'none', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)', backgroundColor: 'var(--tooltip-bg, #fff)', color: 'var(--tooltip-text, #000)' }}
                  />
                  <Area type="monotone" dataKey="intensity" stroke="#10b981" strokeWidth={3} fillOpacity={1} fill="url(#colorIntensity)" />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex flex-col items-center justify-center text-slate-400 italic">
                <History size={48} className="mb-4 opacity-20" />
                <p>Log more moods to see your emotional trends.</p>
              </div>
            )}
          </div>

          <div className="mt-8 grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="p-4 bg-slate-50 dark:bg-slate-900/50 rounded-2xl text-center">
              <p className="text-xs text-slate-500 dark:text-slate-400 font-bold uppercase mb-1">Avg Intensity</p>
              <p className="text-2xl font-bold text-slate-800 dark:text-slate-100">
                {history.length > 0 ? (history.reduce((acc, curr) => acc + curr.intensity, 0) / history.length).toFixed(1) : '-'}
              </p>
            </div>
            <div className="p-4 bg-slate-50 dark:bg-slate-900/50 rounded-2xl text-center">
              <p className="text-xs text-slate-500 dark:text-slate-400 font-bold uppercase mb-1">Most Frequent</p>
              <p className="text-2xl font-bold text-slate-800 dark:text-slate-100 capitalize">
                {history.length > 0 ? 'Calm' : '-'}
              </p>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
};

export default MoodTracker;
