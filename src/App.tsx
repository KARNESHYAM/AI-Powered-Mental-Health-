import React, { useState, useEffect, createContext, useContext } from 'react';
import { 
  doc, 
  getDoc, 
  setDoc, 
  onSnapshot,
  serverTimestamp 
} from 'firebase/firestore';
import { db, auth, googleProvider, signInWithPopup, signOut, onAuthStateChanged, handleFirestoreError, OperationType } from './firebase';
import { UserProfile } from './types';
import { 
  MessageCircle, 
  LayoutDashboard, 
  Smile, 
  BookOpen, 
  Wind, 
  AlertCircle, 
  Settings, 
  LogOut,
  Menu,
  X,
  Heart
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';

// Components
import Dashboard from './components/Dashboard';
import Chat from './components/Chat';
import MoodTracker from './components/MoodTracker';
import Journal from './components/Journal';
import Exercises from './components/Exercises';
import PanicButton from './components/PanicButton';
import SettingsPage from './components/Settings';
import LandingPage from './components/LandingPage';
import LoginPage from './components/LoginPage';

// Context
interface AuthContextType {
  user: any | null;
  profile: UserProfile | null;
  loading: boolean;
  isSigningIn: boolean;
  signInWithGoogle: () => Promise<void>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) throw new Error('useAuth must be used within an AuthProvider');
  return context;
};

const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<any | null>(null);
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [loading, setLoading] = useState(true);
  const [isSigningIn, setIsSigningIn] = useState(false);

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (firebaseUser) => {
      setUser(firebaseUser);
      if (firebaseUser) {
        const docRef = doc(db, 'users', firebaseUser.uid);
        
        // Real-time listener for profile changes
        const unsubscribeProfile = onSnapshot(docRef, (doc) => {
          if (doc.exists()) {
            setProfile(doc.data() as UserProfile);
          } else {
            // Create profile if it doesn't exist
            const newProfile: UserProfile = {
              uid: firebaseUser.uid,
              email: firebaseUser.email || '',
              displayName: firebaseUser.displayName || '',
              photoURL: firebaseUser.photoURL || '',
              createdAt: new Date().toISOString(),
              lastLogin: new Date().toISOString(),
              settings: {
                darkMode: false,
                notifications: true
              },
              role: 'user'
            };
            setDoc(docRef, newProfile).catch(err => handleFirestoreError(err, OperationType.WRITE, `users/${firebaseUser.uid}`));
          }
          setLoading(false);
        }, (error) => {
          handleFirestoreError(error, OperationType.GET, `users/${firebaseUser.uid}`);
          setLoading(false);
        });

        return () => unsubscribeProfile();
      } else {
        setProfile(null);
        setLoading(false);
      }
    });

    return () => unsubscribe();
  }, []);

  const signInWithGoogle = async () => {
    if (isSigningIn) return;
    
    setIsSigningIn(true);
    try {
      await signInWithPopup(auth, googleProvider);
    } catch (error: any) {
      console.error("Sign in error:", error);
      throw error;
    } finally {
      setIsSigningIn(false);
    }
  };

  const logout = async () => {
    try {
      await signOut(auth);
    } catch (error) {
      console.error("Logout error:", error);
    }
  };

  return (
    <AuthContext.Provider value={{ user, profile, loading, isSigningIn, signInWithGoogle, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

const AppContent: React.FC = () => {
  const { user, profile, loading, logout } = useAuth();
  const [activeTab, setActiveTab] = useState('dashboard');
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [showWelcome, setShowWelcome] = useState(false);
  const [showLogin, setShowLogin] = useState(false);

  useEffect(() => {
    const hasSeenWelcome = localStorage.getItem('hasSeenWelcome');
    if (user && !hasSeenWelcome) {
      setShowWelcome(true);
      localStorage.setItem('hasSeenWelcome', 'true');
    }
  }, [user]);

  useEffect(() => {
    if (profile?.settings.darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [profile?.settings.darkMode]);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-50">
        <motion.div 
          animate={{ scale: [1, 1.1, 1] }}
          transition={{ repeat: Infinity, duration: 2 }}
          className="text-emerald-600"
        >
          <Heart size={48} fill="currentColor" />
        </motion.div>
      </div>
    );
  }

  if (!user) {
    if (showLogin) {
      return <LoginPage onBack={() => setShowLogin(false)} />;
    }
    return <LandingPage onStart={() => setShowLogin(true)} />;
  }

  const renderContent = () => {
    switch (activeTab) {
      case 'dashboard': return <Dashboard onNavigate={setActiveTab} />;
      case 'chat': return <Chat />;
      case 'mood': return <MoodTracker />;
      case 'journal': return <Journal />;
      case 'exercises': return <Exercises />;
      case 'panic': return <PanicButton />;
      case 'settings': return <SettingsPage />;
      default: return <Dashboard onNavigate={setActiveTab} />;
    }
  };

  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'chat', label: 'AI Therapist', icon: MessageCircle },
    { id: 'mood', label: 'Mood Tracker', icon: Smile },
    { id: 'journal', label: 'Journal', icon: BookOpen },
    { id: 'exercises', label: 'Exercises', icon: Wind },
    { id: 'panic', label: 'Panic Button', icon: AlertCircle, color: 'text-rose-500' },
    { id: 'settings', label: 'Settings', icon: Settings },
  ];

  return (
    <div className="min-h-screen bg-slate-50 flex relative overflow-hidden">
      {/* Watermark Background */}
      <div className="fixed inset-0 pointer-events-none opacity-[0.03] z-0">
        <img 
          src="https://images.unsplash.com/photo-1502082553048-f009c37129b9?auto=format&fit=crop&w=1000&q=80" 
          alt="Watermark" 
          className="w-full h-full object-cover grayscale"
          referrerPolicy="no-referrer"
        />
      </div>

      {/* Sidebar - Desktop */}
      <aside className="hidden md:flex flex-col w-64 bg-white border-r border-slate-200 p-6">
        <div className="flex items-center gap-3 mb-10 px-2">
          <Heart className="text-emerald-600" size={28} fill="currentColor" />
          <span className="font-bold text-xl text-slate-800">AI Therapy Companion</span>
        </div>
        
        <nav className="flex-1 space-y-2">
          {navItems.map((item) => (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all ${
                activeTab === item.id 
                  ? 'bg-emerald-50 text-emerald-700 font-semibold' 
                  : 'text-slate-500 hover:bg-slate-50 hover:text-slate-700'
              } ${item.color || ''}`}
            >
              <item.icon size={20} />
              {item.label}
            </button>
          ))}
        </nav>

        <div className="mt-auto pt-6 border-t border-slate-100">
          <div className="flex items-center gap-3 px-2 mb-4">
            {profile?.photoURL ? (
              <img src={profile.photoURL} className="w-10 h-10 rounded-full border-2 border-emerald-100" alt="Avatar" />
            ) : (
              <div className="w-10 h-10 rounded-full bg-emerald-100 flex items-center justify-center text-emerald-600 font-bold border-2 border-emerald-50">
                {profile?.displayName?.charAt(0).toUpperCase() || 'U'}
              </div>
            )}
            <div className="overflow-hidden">
              <p className="text-sm font-semibold text-slate-800 truncate">{profile?.displayName || 'User'}</p>
              <p className="text-xs text-slate-500 truncate">{profile?.email || 'Anonymous'}</p>
            </div>
          </div>
          <button 
            onClick={logout}
            className="w-full flex items-center gap-3 px-4 py-3 rounded-xl text-slate-500 hover:bg-rose-50 hover:text-rose-600 transition-all"
          >
            <LogOut size={20} />
            Sign Out
          </button>
        </div>
      </aside>

      {/* Mobile Header */}
      <div className="md:hidden fixed top-0 left-0 right-0 bg-white border-b border-slate-200 h-16 flex items-center justify-between px-4 z-50">
        <button onClick={() => setIsSidebarOpen(true)} className="p-2 text-slate-600">
          <Menu size={24} />
        </button>
        <div className="flex items-center gap-2">
          <Heart className="text-emerald-600" size={24} fill="currentColor" />
          <span className="font-bold text-lg text-slate-800">AI Therapy Companion</span>
        </div>
        <div className="w-10 h-10 flex items-center justify-center">
          {profile?.photoURL ? (
            <img src={profile.photoURL} className="w-8 h-8 rounded-full ml-auto" alt="Avatar" />
          ) : (
            <div className="w-8 h-8 rounded-full bg-emerald-100 flex items-center justify-center text-emerald-600 font-bold text-xs ml-auto">
              {profile?.displayName?.charAt(0).toUpperCase() || 'U'}
            </div>
          )}
        </div>
      </div>

      {/* Mobile Sidebar Overlay */}
      <AnimatePresence>
        {isSidebarOpen && (
          <>
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setIsSidebarOpen(false)}
              className="fixed inset-0 bg-black/20 backdrop-blur-sm z-[60] md:hidden"
            />
            <motion.aside
              initial={{ x: '-100%' }}
              animate={{ x: 0 }}
              exit={{ x: '-100%' }}
              transition={{ type: 'spring', damping: 25, stiffness: 200 }}
              className="fixed inset-y-0 left-0 w-72 bg-white z-[70] p-6 flex flex-col md:hidden"
            >
              <div className="flex items-center justify-between mb-10">
                <div className="flex items-center gap-3">
                  <Heart className="text-emerald-600" size={28} fill="currentColor" />
                  <span className="font-bold text-xl text-slate-800">AI Therapy Companion</span>
                </div>
                <button onClick={() => setIsSidebarOpen(false)} className="p-2 text-slate-400">
                  <X size={24} />
                </button>
              </div>

              <nav className="flex-1 space-y-2">
                {navItems.map((item) => (
                  <button
                    key={item.id}
                    onClick={() => {
                      setActiveTab(item.id);
                      setIsSidebarOpen(false);
                    }}
                    className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all ${
                      activeTab === item.id 
                        ? 'bg-emerald-50 text-emerald-700 font-semibold' 
                        : 'text-slate-500 hover:bg-slate-50 hover:text-slate-700'
                    } ${item.color || ''}`}
                  >
                    <item.icon size={20} />
                    {item.label}
                  </button>
                ))}
              </nav>

              <div className="mt-auto pt-6 border-t border-slate-100">
                <button 
                  onClick={() => {
                    setIsSidebarOpen(false);
                    logout();
                  }}
                  className="w-full flex items-center gap-3 px-4 py-3 rounded-xl text-slate-500 hover:bg-rose-50 hover:text-rose-600 transition-all"
                >
                  <LogOut size={20} />
                  Sign Out
                </button>
              </div>
            </motion.aside>
          </>
        )}
      </AnimatePresence>

      {/* Main Content */}
      <main className="flex-1 md:h-screen overflow-y-auto pt-16 md:pt-0">
        <div className="max-w-6xl mx-auto p-4 md:p-8">
          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, x: 10 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -10 }}
              transition={{ duration: 0.2 }}
            >
              {renderContent()}
            </motion.div>
          </AnimatePresence>
        </div>
      </main>

      {/* Welcome Dialog */}
      <AnimatePresence>
        {showWelcome && (
          <div className="fixed inset-0 flex items-center justify-center z-[100] p-4">
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setShowWelcome(false)}
              className="absolute inset-0 bg-slate-900/40 backdrop-blur-md"
            />
            <motion.div
              initial={{ opacity: 0, scale: 0.9, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.9, y: 20 }}
              className="relative bg-white p-10 rounded-[2.5rem] shadow-2xl max-w-md w-full text-center overflow-hidden"
            >
              <div className="absolute top-0 left-0 w-full h-2 bg-gradient-to-r from-emerald-400 to-teal-500" />
              <div className="w-20 h-20 bg-emerald-50 rounded-3xl flex items-center justify-center text-emerald-600 mx-auto mb-6">
                <Heart size={40} fill="currentColor" />
              </div>
              <h2 className="text-3xl font-bold text-slate-800 mb-4">Welcome, {profile?.displayName?.split(' ')[0] || 'Friend'}!</h2>
              <p className="text-slate-500 mb-8 leading-relaxed">
                We're so glad you're here. This is your safe space for reflection, growth, and support. Take a deep breath and explore your companion.
              </p>
              <button
                onClick={() => setShowWelcome(false)}
                className="w-full py-4 bg-emerald-600 text-white rounded-2xl font-bold shadow-lg shadow-emerald-100 hover:bg-emerald-700 transition-all"
              >
                Let's Begin
              </button>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default function App() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
}
