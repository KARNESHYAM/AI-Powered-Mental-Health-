import React, { useState, useEffect } from 'react';
import { collection, addDoc, serverTimestamp, query, where, orderBy, onSnapshot, deleteDoc, doc } from 'firebase/firestore';
import { db, handleFirestoreError, OperationType } from '../firebase';
import { useAuth } from '../App';
import { JournalEntry } from '../types';
import { getJournalInsights } from '../services/aiService';
import { motion, AnimatePresence } from 'motion/react';
import { BookOpen, Plus, Search, Trash2, Sparkles, Calendar, Tag, ChevronRight } from 'lucide-react';
import { format } from 'date-fns';
import ConfirmModal from './ConfirmModal';

const Journal: React.FC = () => {
  const { user } = useAuth();
  const [entries, setEntries] = useState<JournalEntry[]>([]);
  const [isAdding, setIsAdding] = useState(false);
  const [search, setSearch] = useState('');
  const [selectedEntry, setSelectedEntry] = useState<JournalEntry | null>(null);
  const [entryToDelete, setEntryToDelete] = useState<string | null>(null);

  // Form state
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    if (!user) return;

    const q = query(
      collection(db, 'journals'),
      where('uid', '==', user.uid),
      orderBy('timestamp', 'desc')
    );

    const unsubscribe = onSnapshot(q, (snapshot) => {
      setEntries(snapshot.docs.map(doc => ({ id: doc.id, ...doc.data() } as JournalEntry)));
    }, (error) => {
      handleFirestoreError(error, OperationType.LIST, 'journals');
    });

    return unsubscribe;
  }, [user]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!user || !content.trim()) return;
    setIsSubmitting(true);

    try {
      // Get AI insights
      const { insight, tags } = await getJournalInsights(content);

      await addDoc(collection(db, 'journals'), {
        uid: user.uid,
        title: title || 'Untitled Entry',
        content,
        tags: tags || [],
        aiInsights: insight || '',
        timestamp: serverTimestamp()
      });

      setTitle('');
      setContent('');
      setIsAdding(false);
    } catch (error) {
      handleFirestoreError(error, OperationType.CREATE, 'journals');
    } finally {
      setIsSubmitting(false);
    }
  };

  const deleteEntry = async (id: string) => {
    try {
      await deleteDoc(doc(db, 'journals', id));
      if (selectedEntry?.id === id) setSelectedEntry(null);
      setEntryToDelete(null);
    } catch (error) {
      handleFirestoreError(error, OperationType.DELETE, `journals/${id}`);
    }
  };

  const filteredEntries = entries.filter(e => 
    e.title.toLowerCase().includes(search.toLowerCase()) || 
    e.content.toLowerCase().includes(search.toLowerCase()) ||
    e.tags.some(t => t.toLowerCase().includes(search.toLowerCase()))
  );

  return (
    <div className="space-y-8">
      <header className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-slate-800">Personal Journal</h1>
          <p className="text-slate-500">Write your thoughts and let AI help you reflect.</p>
        </div>
        <button 
          onClick={() => setIsAdding(true)}
          className="bg-emerald-600 text-white px-6 py-3 rounded-2xl font-bold flex items-center justify-center gap-2 hover:bg-emerald-700 transition-all shadow-lg shadow-emerald-100"
        >
          <Plus size={20} /> New Entry
        </button>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Sidebar / List */}
        <div className="lg:col-span-4 space-y-4">
          <div className="relative">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
            <input 
              type="text" 
              placeholder="Search entries..." 
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full bg-white border border-slate-100 rounded-2xl pl-12 pr-4 py-3 text-sm focus:ring-2 focus:ring-emerald-500 transition-all shadow-sm"
            />
          </div>

          <div className="space-y-3 max-h-[600px] overflow-y-auto pr-2 custom-scrollbar">
            {filteredEntries.map((entry) => (
              <button
                key={entry.id}
                onClick={() => {
                  setSelectedEntry(entry);
                  setIsAdding(false);
                }}
                className={`w-full text-left p-5 rounded-3xl transition-all border ${
                  selectedEntry?.id === entry.id 
                    ? 'bg-emerald-50 border-emerald-200 shadow-sm' 
                    : 'bg-white border-slate-100 hover:border-emerald-100 hover:bg-slate-50'
                }`}
              >
                <div className="flex justify-between items-start mb-2">
                  <h3 className="font-bold text-slate-800 truncate pr-2">{entry.title}</h3>
                  <span className="text-[10px] text-slate-400 font-bold uppercase whitespace-nowrap">
                    {entry.timestamp?.toDate ? format(entry.timestamp.toDate(), 'MMM d') : 'Today'}
                  </span>
                </div>
                <p className="text-xs text-slate-500 line-clamp-2 mb-3">{entry.content}</p>
                <div className="flex flex-wrap gap-1">
                  {entry.tags.slice(0, 3).map(tag => (
                    <span key={tag} className="text-[9px] px-2 py-0.5 bg-slate-100 text-slate-600 rounded-full font-bold uppercase tracking-wider">
                      {tag}
                    </span>
                  ))}
                </div>
              </button>
            ))}
            {filteredEntries.length === 0 && (
              <div className="text-center py-12 text-slate-400 italic">
                No entries found.
              </div>
            )}
          </div>
        </div>

        {/* Content Area */}
        <div className="lg:col-span-8">
          <AnimatePresence mode="wait">
            {isAdding ? (
              <motion.div
                key="add"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                className="bg-white p-8 rounded-3xl shadow-sm border border-slate-100"
              >
                <div className="flex items-center justify-between mb-8">
                  <h2 className="text-2xl font-bold text-slate-800">Write New Entry</h2>
                  <button onClick={() => setIsAdding(false)} className="text-slate-400 hover:text-slate-600">
                    Cancel
                  </button>
                </div>
                <form onSubmit={handleSubmit} className="space-y-6">
                  <input 
                    type="text" 
                    placeholder="Title" 
                    value={title}
                    onChange={(e) => setTitle(e.target.value)}
                    className="w-full text-2xl font-bold border-none focus:ring-0 p-0 placeholder:text-slate-300"
                  />
                  <textarea 
                    placeholder="Start writing..." 
                    value={content}
                    onChange={(e) => setContent(e.target.value)}
                    className="w-full min-h-[400px] border-none focus:ring-0 p-0 text-slate-700 leading-relaxed resize-none placeholder:text-slate-300"
                  />
                  <div className="pt-6 border-t border-slate-100 flex justify-end">
                    <button 
                      type="submit"
                      disabled={isSubmitting || !content.trim()}
                      className="bg-emerald-600 text-white px-8 py-4 rounded-2xl font-bold hover:bg-emerald-700 transition-all shadow-lg shadow-emerald-100 disabled:opacity-50 flex items-center gap-2"
                    >
                      {isSubmitting ? 'Analyzing...' : <><Sparkles size={20} /> Save & Get Insights</>}
                    </button>
                  </div>
                </form>
              </motion.div>
            ) : selectedEntry ? (
              <motion.div
                key={selectedEntry.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="space-y-6"
              >
                <div className="bg-white p-8 rounded-3xl shadow-sm border border-slate-100">
                  <div className="flex justify-between items-start mb-6">
                    <div>
                      <h2 className="text-3xl font-bold text-slate-800 mb-2">{selectedEntry.title}</h2>
                      <div className="flex items-center gap-4 text-sm text-slate-400">
                        <span className="flex items-center gap-1"><Calendar size={14} /> {selectedEntry.timestamp?.toDate ? format(selectedEntry.timestamp.toDate(), 'MMMM d, yyyy') : 'Today'}</span>
                        <span className="flex items-center gap-1"><Tag size={14} /> {selectedEntry.tags.join(', ')}</span>
                      </div>
                    </div>
                    <button 
                      onClick={() => setEntryToDelete(selectedEntry.id!)}
                      className="p-2 text-slate-300 hover:text-rose-500 transition-colors"
                    >
                      <Trash2 size={20} />
                    </button>
                  </div>
                  <div className="prose prose-slate max-w-none text-slate-700 leading-relaxed whitespace-pre-wrap">
                    {selectedEntry.content}
                  </div>
                </div>

                {selectedEntry.aiInsights && (
                  <motion.div 
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    className="bg-gradient-to-br from-blue-500 to-indigo-600 p-8 rounded-3xl text-white shadow-lg shadow-blue-100 relative overflow-hidden"
                  >
                    <Sparkles className="absolute -right-4 -top-4 text-white opacity-10" size={120} />
                    <h3 className="text-xl font-bold mb-3 flex items-center gap-2">
                      <Sparkles size={20} /> AI Insight
                    </h3>
                    <p className="text-blue-50 leading-relaxed relative z-10">
                      {selectedEntry.aiInsights}
                    </p>
                  </motion.div>
                )}
              </motion.div>
            ) : (
              <div className="h-full flex flex-col items-center justify-center text-center py-20 bg-white rounded-3xl border border-dashed border-slate-200">
                <div className="w-20 h-20 bg-slate-50 rounded-full flex items-center justify-center text-slate-300 mb-6">
                  <BookOpen size={40} />
                </div>
                <h3 className="text-xl font-bold text-slate-800 mb-2">Select an entry to read</h3>
                <p className="text-slate-500 max-w-xs">Your thoughts are safe here. Select an entry from the list or start a new one.</p>
              </div>
            )}
          </AnimatePresence>
        </div>
      </div>
      <ConfirmModal 
        isOpen={!!entryToDelete}
        title="Delete Journal Entry?"
        message="This will permanently remove this entry. This action cannot be undone."
        confirmLabel="Delete"
        onConfirm={() => entryToDelete && deleteEntry(entryToDelete)}
        onCancel={() => setEntryToDelete(null)}
      />
    </div>
  );
};

export default Journal;
