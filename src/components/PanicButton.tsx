import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { AlertCircle, Wind, Heart, ShieldCheck, ChevronRight, X } from 'lucide-react';

const PanicButton: React.FC = () => {
  const [isPanicMode, setIsPanicMode] = useState(false);
  const [step, setStep] = useState(0);

  const steps = [
    {
      title: "Just Breathe",
      desc: "You are safe. Everything is okay. Let's take a deep breath together.",
      action: "Start Breathing Animation"
    },
    {
      title: "Acknowledge",
      desc: "What you're feeling is a physical reaction. It will pass. It always does.",
      action: "Next"
    },
    {
      title: "Grounding",
      desc: "Press your feet firmly into the floor. Feel the support beneath you.",
      action: "Next"
    },
    {
      title: "Affirmation",
      desc: "I am strong. I am capable. I am in control of my breath.",
      action: "Finish"
    }
  ];

  return (
    <div className="h-full flex flex-col items-center justify-center py-12">
      <AnimatePresence mode="wait">
        {!isPanicMode ? (
          <motion.div
            key="idle"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            className="text-center space-y-8"
          >
            <div className="relative">
              <motion.div 
                animate={{ scale: [1, 1.2, 1], opacity: [0.2, 0.4, 0.2] }}
                transition={{ repeat: Infinity, duration: 3 }}
                className="absolute inset-0 bg-rose-500 rounded-full blur-3xl"
              />
              <button
                onClick={() => setIsPanicMode(true)}
                className="relative w-48 h-48 bg-rose-500 text-white rounded-full shadow-2xl shadow-rose-200 flex flex-col items-center justify-center gap-2 hover:bg-rose-600 transition-all active:scale-95 z-10"
              >
                <AlertCircle size={48} />
                <span className="font-bold text-xl uppercase tracking-widest">Help</span>
              </button>
            </div>
            
            <div className="max-w-sm">
              <h2 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">Feeling Overwhelmed?</h2>
              <p className="text-slate-500 dark:text-slate-400">Press the button for instant calming techniques and guidance through a panic attack.</p>
            </div>
          </motion.div>
        ) : (
          <motion.div
            key="panic"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-white dark:bg-slate-900 z-[100] flex flex-col items-center justify-center p-6"
          >
            <button 
              onClick={() => { setIsPanicMode(false); setStep(0); }}
              className="absolute top-8 right-8 p-3 text-slate-400 hover:text-slate-600 bg-slate-50 dark:bg-slate-800 rounded-full"
            >
              <X size={24} />
            </button>

            <div className="max-w-lg w-full text-center space-y-12">
              <div className="relative flex items-center justify-center h-64">
                <motion.div
                  animate={{ scale: [1, 1.8, 1], opacity: [0.3, 0.6, 0.3] }}
                  transition={{ repeat: Infinity, duration: 6, ease: "easeInOut" }}
                  className="absolute w-32 h-32 bg-emerald-400 rounded-full blur-3xl"
                />
                <div className="relative z-10 flex flex-col items-center">
                  <div className="w-24 h-24 bg-emerald-100 dark:bg-emerald-900/30 text-emerald-600 dark:text-emerald-400 rounded-full flex items-center justify-center mb-6">
                    <Wind size={40} />
                  </div>
                  <p className="text-emerald-600 dark:text-emerald-400 font-bold uppercase tracking-[0.3em]">Breathe</p>
                </div>
              </div>

              <AnimatePresence mode="wait">
                <motion.div
                  key={step}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className="space-y-4"
                >
                  <h2 className="text-4xl font-bold text-slate-800 dark:text-slate-100">{steps[step].title}</h2>
                  <p className="text-xl text-slate-600 dark:text-slate-300 leading-relaxed">{steps[step].desc}</p>
                </motion.div>
              </AnimatePresence>

              <div className="pt-8">
                <button
                  onClick={() => {
                    if (step < steps.length - 1) setStep(step + 1);
                    else { setIsPanicMode(false); setStep(0); }
                  }}
                  className="px-12 py-5 bg-slate-900 dark:bg-slate-100 text-white dark:text-slate-900 rounded-2xl font-bold text-lg hover:bg-slate-800 dark:hover:bg-white transition-all shadow-xl flex items-center gap-2 mx-auto"
                >
                  {steps[step].action} <ChevronRight size={20} />
                </button>
              </div>

              <div className="flex justify-center gap-2">
                {steps.map((_, i) => (
                  <div 
                    key={i} 
                    className={`h-1.5 rounded-full transition-all ${i === step ? 'w-8 bg-slate-900 dark:bg-slate-100' : 'w-2 bg-slate-200 dark:bg-slate-700'}`} 
                  />
                ))}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default PanicButton;
