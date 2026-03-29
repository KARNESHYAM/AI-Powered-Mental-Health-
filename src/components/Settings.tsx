import React, { useState } from 'react';
import { useAuth } from '../App';
import { db, handleFirestoreError, OperationType } from '../firebase';
import { doc, updateDoc, deleteDoc } from 'firebase/firestore';
import { motion, AnimatePresence } from 'motion/react';
import { User, Info, Mail, Trash2, ChevronRight, AlertTriangle } from 'lucide-react';
import { collection, getDocs, writeBatch, query, where } from 'firebase/firestore';

const Settings: React.FC = () => {
  const { user, profile, logout } = useAuth();
  const [isDeleting, setIsDeleting] = useState(false);

  const deleteAccount = async () => {
    if (!user) return;
    
    const confirm = window.confirm(
      "CRITICAL: This will permanently delete your account and all your data (moods, journals, chats). This action cannot be undone. Proceed?"
    );
    
    if (!confirm) return;

    setIsDeleting(true);
    try {
      const batch = writeBatch(db);
      
      // 1. Delete Moods
      const moodsQ = query(collection(db, 'moods'), where('uid', '==', user.uid));
      const moodsSnap = await getDocs(moodsQ);
      moodsSnap.docs.forEach(d => batch.delete(d.ref));

      // 2. Delete Journals
      const journalsQ = query(collection(db, 'journals'), where('uid', '==', user.uid));
      const journalsSnap = await getDocs(journalsQ);
      journalsSnap.docs.forEach(d => batch.delete(d.ref));

      // 3. Delete Chat Sessions and their Messages
      const sessionsQ = query(collection(db, 'chatSessions'), where('uid', '==', user.uid));
      const sessionsSnap = await getDocs(sessionsQ);
      
      for (const sessionDoc of sessionsSnap.docs) {
        // Delete subcollection messages
        const messagesSnap = await getDocs(collection(db, 'chatSessions', sessionDoc.id, 'messages'));
        messagesSnap.docs.forEach(m => batch.delete(m.ref));
        // Delete session
        batch.delete(sessionDoc.ref);
      }

      // 4. Delete user profile
      batch.delete(doc(db, 'users', user.uid));

      await batch.commit();
      
      await logout();
      window.location.reload();
    } catch (error) {
      console.error("Error deleting account:", error);
      alert("Failed to delete account. You may need to re-authenticate (log out and log in again) to perform this sensitive action.");
    } finally {
      setIsDeleting(false);
    }
  };

  return (
    <div className="space-y-8 max-w-2xl mx-auto">
      <header>
        <h1 className="text-3xl font-bold text-slate-800">Account & Info</h1>
        <p className="text-slate-500">Manage your account and learn more about the app.</p>
      </header>

      <div className="space-y-6">
        {/* Profile Section */}
        <section className="bg-white p-8 rounded-3xl shadow-sm border border-slate-100">
          <h2 className="text-lg font-bold text-slate-800 mb-6 flex items-center gap-2">
            <User size={20} className="text-emerald-500" /> Account
          </h2>
          <div className="flex items-center gap-6">
            <div className="w-20 h-20 rounded-full border-4 border-emerald-50 bg-emerald-100 flex items-center justify-center text-emerald-600 text-2xl font-bold">
              {profile?.displayName?.charAt(0).toUpperCase() || 'U'}
            </div>
            <div className="flex-1">
              <p className="font-bold text-slate-800 text-lg">{profile?.displayName || 'User'}</p>
              <p className="text-slate-500">{profile?.email || 'Anonymous'}</p>
              <p className="text-xs text-slate-400 mt-1">Phone: {profile?.phoneNumber || 'Not provided'}</p>
            </div>
          </div>
        </section>

        {/* App Info */}
        <section className="bg-white p-8 rounded-3xl shadow-sm border border-slate-100">
          <h2 className="text-lg font-bold text-slate-800 mb-6 flex items-center gap-2">
            <Info size={20} className="text-blue-500" /> About the App
          </h2>
          <div className="prose prose-slate prose-sm max-w-none text-slate-600">
            <p>
              <strong>AI Therapy Companion</strong> is your personal space for mental wellness. We combine modern AI technology with proven therapeutic techniques to help you navigate life's challenges.
            </p>
            <p>
              Our mission is to make emotional support accessible to everyone, anywhere, at any time. Whether you're feeling anxious, sad, or just need someone to talk to, your companion is always here.
            </p>
            <div className="mt-4 p-4 bg-blue-50 rounded-2xl border border-blue-100">
              <p className="font-bold text-blue-800 mb-1">Version 1.2.0</p>
              <p className="text-blue-600 text-xs">Built with love for mental health awareness.</p>
            </div>
          </div>
        </section>

        {/* Contact Details */}
        <section className="bg-white p-8 rounded-3xl shadow-sm border border-slate-100">
          <h2 className="text-lg font-bold text-slate-800 mb-6 flex items-center gap-2">
            <Mail size={20} className="text-purple-500" /> Contact Support
          </h2>
          <div className="space-y-4">
            <p className="text-slate-600 text-sm">Have questions or need assistance? Our team is here to help you.</p>
            <div className="flex flex-col gap-3">
              <a href="mailto:support@aitherapy.com" className="flex items-center justify-between p-4 bg-slate-50 rounded-2xl hover:bg-slate-100 transition-all group">
                <div className="flex items-center gap-3">
                  <Mail size={18} className="text-slate-400" />
                  <span className="font-semibold text-slate-700">support@aitherapy.com</span>
                </div>
                <ChevronRight size={18} className="text-slate-300 group-hover:translate-x-1 transition-transform" />
              </a>
            </div>
          </div>
        </section>

        {/* Danger Zone */}
        <section className="bg-white p-8 rounded-3xl shadow-sm border border-rose-100">
          <h2 className="text-lg font-bold text-rose-600 mb-6 flex items-center gap-2">
            <AlertTriangle size={20} /> Danger Zone
          </h2>
          <p className="text-slate-500 text-sm mb-6">
            Once you delete your account, there is no going back. Please be certain.
          </p>
          <button 
            onClick={deleteAccount}
            disabled={isDeleting}
            className="w-full flex items-center justify-center gap-3 p-4 bg-rose-50 text-rose-600 hover:bg-rose-600 hover:text-white rounded-2xl font-bold transition-all disabled:opacity-50"
          >
            <Trash2 size={18} />
            {isDeleting ? 'Deleting Data...' : 'Delete My Account & Data'}
          </button>
        </section>
      </div>
    </div>
  );
};

export default Settings;
