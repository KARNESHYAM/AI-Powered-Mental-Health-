import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Wind, Timer, Heart, Brain, ChevronRight, Play, Pause, RotateCcw, ArrowLeft } from 'lucide-react';

const Exercises: React.FC = () => {
  const [activeExercise, setActiveExercise] = useState<string | null>(null);
  const [timer, setTimer] = useState(0);
  const [isActive, setIsActive] = useState(false);
  const [phase, setPhase] = useState<'inhale' | 'hold' | 'exhale' | 'hold-out'>('inhale');

  // Breathing logic
  useEffect(() => {
    let interval: any;
    if (isActive && activeExercise === 'breathing') {
      interval = setInterval(() => {
        setTimer(t => {
          // 4-7-8 Breathing
          if (phase === 'inhale' && t >= 4) { setPhase('hold'); return 0; }
          if (phase === 'hold' && t >= 7) { setPhase('exhale'); return 0; }
          if (phase === 'exhale' && t >= 8) { setPhase('inhale'); return 0; }
          return t + 1;
        });
      }, 1000);
    } else if (isActive && activeExercise === 'meditation') {
      interval = setInterval(() => {
        setTimer(t => t + 1);
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [isActive, phase, activeExercise]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const exercises = [
    {
      id: 'breathing',
      title: '4-7-8 Breathing',
      desc: 'A natural tranquilizer for the nervous system.',
      icon: Wind,
      color: 'bg-emerald-100 text-emerald-600',
      gradient: 'from-emerald-500 to-teal-600'
    },
    {
      id: 'meditation',
      title: 'Mindfulness Timer',
      desc: 'Simple meditation to center your thoughts.',
      icon: Brain,
      color: 'bg-blue-100 text-blue-600',
      gradient: 'from-blue-500 to-indigo-600'
    },
    {
      id: 'grounding',
      title: '5-4-3-2-1 Technique',
      desc: 'Ground yourself in the present moment.',
      icon: Heart,
      color: 'bg-purple-100 text-purple-600',
      gradient: 'from-purple-500 to-fuchsia-600'
    }
  ];

  return (
    <div className="space-y-8">
      <header>
        <h1 className="text-3xl font-bold text-slate-800">Guided Exercises</h1>
        <p className="text-slate-500">Simple techniques to help you find your calm.</p>
      </header>

      <AnimatePresence mode="wait">
        {!activeExercise ? (
          <motion.div 
            key="list"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="grid grid-cols-1 md:grid-cols-3 gap-6"
          >
            {exercises.map((ex) => (
              <button
                key={ex.id}
                onClick={() => {
                  setActiveExercise(ex.id);
                  setTimer(0);
                  setIsActive(false);
                  setPhase('inhale');
                }}
                className="bg-white p-8 rounded-3xl shadow-sm border border-slate-100 text-left hover:shadow-md transition-all group"
              >
                <div className={`w-14 h-14 ${ex.color} rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform`}>
                  <ex.icon size={28} />
                </div>
                <h3 className="text-xl font-bold text-slate-800 mb-2">{ex.title}</h3>
                <p className="text-sm text-slate-500 mb-6">{ex.desc}</p>
                <div className="flex items-center text-emerald-600 font-bold text-sm">
                  Start Now <ChevronRight size={16} />
                </div>
              </button>
            ))}
          </motion.div>
        ) : (
          <motion.div
            key="active"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            className="max-w-2xl mx-auto bg-white p-12 rounded-[3rem] shadow-xl border border-slate-100 text-center"
          >
            <button 
              onClick={() => setActiveExercise(null)}
              className="absolute top-8 left-8 flex items-center gap-2 text-slate-400 hover:text-slate-600 font-bold transition-colors"
            >
              <ArrowLeft size={20} /> Back
            </button>

            <button 
              onClick={() => setActiveExercise(null)}
              className="absolute top-8 right-8 text-slate-400 hover:text-slate-600"
            >
              Close
            </button>

            {activeExercise === 'breathing' && (
              <div className="space-y-12">
                <div className="relative flex items-center justify-center h-64">
                  <motion.div
                    animate={{
                      scale: phase === 'inhale' ? 1.5 : phase === 'hold' ? 1.5 : 1,
                      opacity: phase === 'inhale' ? 0.8 : phase === 'hold' ? 1 : 0.5
                    }}
                    transition={{ 
                      duration: phase === 'inhale' ? 4 : phase === 'hold' ? 7 : 8,
                      ease: "easeInOut"
                    }}
                    className="absolute w-40 h-40 bg-emerald-500 rounded-full blur-3xl"
                  />
                  <div className="relative z-10">
                    <h2 className="text-5xl font-bold text-slate-800 capitalize mb-2">{phase}</h2>
                    <p className="text-2xl font-medium text-emerald-600">{timer}s</p>
                  </div>
                </div>
                
                <div className="space-y-4">
                  <h3 className="text-2xl font-bold text-slate-800">4-7-8 Breathing</h3>
                  <p className="text-slate-500 max-w-sm mx-auto">Inhale for 4s, Hold for 7s, Exhale for 8s. Repeat 4 times.</p>
                </div>

                <div className="flex justify-center gap-4">
                  <button 
                    onClick={() => setIsActive(!isActive)}
                    className={`px-8 py-4 rounded-2xl font-bold flex items-center gap-2 transition-all ${
                      isActive ? 'bg-slate-100 text-slate-600' : 'bg-emerald-600 text-white shadow-lg shadow-emerald-100'
                    }`}
                  >
                    {isActive ? <><Pause size={20} /> Pause</> : <><Play size={20} /> Start</>}
                  </button>
                  <button 
                    onClick={() => { setTimer(0); setPhase('inhale'); setIsActive(false); }}
                    className="px-8 py-4 bg-slate-50 text-slate-500 rounded-2xl font-bold hover:bg-slate-100 transition-all"
                  >
                    <RotateCcw size={20} />
                  </button>
                </div>
              </div>
            )}

            {activeExercise === 'meditation' && (
              <div className="space-y-12">
                <div className="flex flex-col items-center justify-center">
                  <div className="text-7xl font-bold text-slate-800 mb-4 tabular-nums">
                    {formatTime(timer)}
                  </div>
                  <p className="text-slate-500 uppercase tracking-widest font-bold text-xs">Meditation in progress</p>
                </div>

                <div className="flex justify-center gap-4">
                  <button 
                    onClick={() => setIsActive(!isActive)}
                    className={`px-8 py-4 rounded-2xl font-bold flex items-center gap-2 transition-all ${
                      isActive ? 'bg-slate-100 text-slate-600' : 'bg-blue-600 text-white shadow-lg shadow-blue-100'
                    }`}
                  >
                    {isActive ? <><Pause size={20} /> Pause</> : <><Play size={20} /> Start</>}
                  </button>
                  <button 
                    onClick={() => { setTimer(0); setIsActive(false); }}
                    className="px-8 py-4 bg-slate-50 text-slate-500 rounded-2xl font-bold hover:bg-slate-100 transition-all"
                  >
                    <RotateCcw size={20} />
                  </button>
                </div>
              </div>
            )}

            {activeExercise === 'grounding' && (
              <div className="space-y-8 text-left max-w-md mx-auto">
                <h2 className="text-3xl font-bold text-slate-800 text-center mb-8">5-4-3-2-1 Grounding</h2>
                <div className="space-y-6">
                  {[
                    { n: 5, t: 'Things you can SEE', c: 'text-emerald-600 bg-emerald-50' },
                    { n: 4, t: 'Things you can TOUCH', c: 'text-blue-600 bg-blue-50' },
                    { n: 3, t: 'Things you can HEAR', c: 'text-purple-600 bg-purple-50' },
                    { n: 2, t: 'Things you can SMELL', c: 'text-orange-600 bg-orange-50' },
                    { n: 1, t: 'Thing you can TASTE', c: 'text-rose-600 bg-rose-50' },
                  ].map((item) => (
                    <div key={item.n} className="flex items-center gap-4">
                      <div className={`w-12 h-12 rounded-2xl flex items-center justify-center font-bold text-xl ${item.c}`}>
                        {item.n}
                      </div>
                      <p className="font-bold text-slate-700">{item.t}</p>
                    </div>
                  ))}
                </div>
                <div className="pt-8 flex justify-center">
                  <button 
                    onClick={() => setActiveExercise(null)}
                    className="px-8 py-4 bg-emerald-600 text-white rounded-2xl font-bold shadow-lg shadow-emerald-100"
                  >
                    I Feel Better
                  </button>
                </div>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default Exercises;
