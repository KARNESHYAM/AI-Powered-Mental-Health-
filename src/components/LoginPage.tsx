import React, { useState } from 'react';
import { motion } from 'motion/react';
import { User, Mail, Phone, Heart, ArrowRight, ChevronLeft } from 'lucide-react';
import { useAuth } from '../App';

interface LoginPageProps {
  onBack: () => void;
}

const LoginPage: React.FC<LoginPageProps> = ({ onBack }) => {
  const { signInWithGoogle, isSigningIn } = useAuth();
  const [error, setError] = useState<string | null>(null);

  const handleGoogleSignIn = async () => {
    setError(null);
    try {
      await signInWithGoogle();
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
          <p className="text-slate-500">Sign in to start your journey.</p>
        </div>

        <div className="space-y-4">
          {error && (
            <motion.p
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              className="text-rose-500 text-sm font-bold text-center mb-4"
            >
              {error}
            </motion.p>
          )}

          <button
            onClick={handleGoogleSignIn}
            disabled={isSigningIn}
            className="w-full py-4 bg-white border-2 border-slate-100 text-slate-700 rounded-2xl font-bold shadow-sm hover:bg-slate-50 transition-all flex items-center justify-center gap-3 group disabled:opacity-50"
          >
            {isSigningIn ? (
              <div className="w-6 h-6 border-2 border-emerald-600/30 border-t-emerald-600 rounded-full animate-spin" />
            ) : (
              <>
                <img src="https://www.gstatic.com/firebasejs/ui/2.0.0/images/auth/google.svg" alt="Google" className="w-5 h-5" />
                Continue with Google
              </>
            )}
          </button>
        </div>
      </motion.div>
    </div>
  );
};

export default LoginPage;
