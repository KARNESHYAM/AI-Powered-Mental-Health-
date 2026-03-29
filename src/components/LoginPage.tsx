import React, { useState } from 'react';
import { motion } from 'motion/react';
import { User, Mail, Phone, Heart, ArrowRight, ChevronLeft } from 'lucide-react';
import { useAuth } from '../App';

interface LoginPageProps {
  onBack: () => void;
}

const LoginPage: React.FC<LoginPageProps> = ({ onBack }) => {
  const { signInCustom, isSigningIn } = useAuth();
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim() || !email.trim()) {
      setError('Please fill in all fields.');
      return;
    }

    setError(null);
    try {
      await signInCustom(name, email);
    } catch (err: any) {
      setError(err.message || 'An error occurred during login.');
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-slate-50 p-4 relative overflow-hidden">
      {/* Decorative Background */}
      <div className="absolute inset-0 pointer-events-none opacity-[0.05] z-0">
        <img 
          src="https://images.unsplash.com/photo-1441974231531-c6227db76b6e?auto=format&fit=crop&w=1000&q=80" 
          alt="Watermark" 
          className="w-full h-full object-cover grayscale"
          referrerPolicy="no-referrer"
        />
      </div>
      <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-emerald-200/20 rounded-full blur-[120px]" />
      <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-blue-200/20 rounded-full blur-[120px]" />

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="relative z-10 w-full max-w-md bg-white p-8 md:p-10 rounded-[2.5rem] shadow-2xl border border-white"
      >
        <button 
          onClick={onBack}
          className="absolute left-6 top-8 p-2 text-slate-400 hover:text-emerald-600 transition-colors"
        >
          <ChevronLeft size={24} />
        </button>

        <div className="text-center mb-8">
          <div className="w-16 h-16 bg-emerald-50 rounded-2xl flex items-center justify-center text-emerald-600 mx-auto mb-6">
            <Heart size={32} fill="currentColor" />
          </div>
          <h2 className="text-3xl font-bold text-slate-800 mb-2">Welcome</h2>
          <p className="text-slate-500">Enter your details to start your journey.</p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-5">
          <div className="space-y-1.5">
            <label className="text-sm font-bold text-slate-700 ml-1">Full Name</label>
            <div className="relative group">
              <User className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400 group-focus-within:text-emerald-500 transition-colors" size={18} />
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="John Doe"
                className="w-full pl-12 pr-4 py-3.5 bg-slate-50 border-none rounded-2xl focus:ring-2 focus:ring-emerald-500 transition-all outline-none"
                required
              />
            </div>
          </div>

          <div className="space-y-1.5">
            <label className="text-sm font-bold text-slate-700 ml-1">Email Address</label>
            <div className="relative group">
              <Mail className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400 group-focus-within:text-emerald-500 transition-colors" size={18} />
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="john@example.com"
                className="w-full pl-12 pr-4 py-3.5 bg-slate-50 border-none rounded-2xl focus:ring-2 focus:ring-emerald-500 transition-all outline-none"
                required
              />
            </div>
          </div>

          {error && (
            <motion.p
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              className="text-rose-500 text-sm font-bold text-center"
            >
              {error}
            </motion.p>
          )}

          <button
            type="submit"
            disabled={isSigningIn}
            className="w-full py-4 bg-emerald-600 text-white rounded-2xl font-bold shadow-xl shadow-emerald-100 hover:bg-emerald-700 transition-all flex items-center justify-center gap-2 group disabled:opacity-50 mt-4"
          >
            {isSigningIn ? (
              <div className="w-6 h-6 border-2 border-white/30 border-t-white rounded-full animate-spin" />
            ) : (
              <>
                Start Journey
                <ArrowRight size={20} className="group-hover:translate-x-1 transition-transform" />
              </>
            )}
          </button>
        </form>
      </motion.div>
    </div>
  );
};

export default LoginPage;
