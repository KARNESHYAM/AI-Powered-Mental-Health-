import React, { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { 
  Heart, 
  MessageCircle, 
  Smile, 
  BookOpen, 
  Wind, 
  AlertCircle, 
  Shield, 
  Sparkles, 
  ChevronRight, 
  HelpCircle, 
  X,
  CheckCircle2,
  Users,
  Zap
} from 'lucide-react';
import { useAuth } from '../App';

interface LandingPageProps {
  onStart: () => void;
}

const LandingPage: React.FC<LandingPageProps> = ({ onStart }) => {
  const { isSigningIn } = useAuth();
  const [selectedFeature, setSelectedFeature] = useState<any>(null);

  const features = [
    {
      icon: MessageCircle,
      title: "AI Therapist",
      desc: "Empathetic, non-judgmental chat support available 24/7 using CBT techniques.",
      longDesc: "Our AI Therapist is trained on thousands of therapeutic interactions to provide high-quality, empathetic support. It uses Cognitive Behavioral Therapy (CBT) frameworks to help you identify negative thought patterns and develop healthier coping mechanisms. Whether it's 3 AM or 3 PM, your companion is always ready to listen.",
      color: "text-emerald-600",
      bg: "bg-emerald-50",
      img: "https://images.unsplash.com/photo-1573497019940-1c28c88b4f3e?auto=format&fit=crop&w=800&q=80"
    },
    {
      icon: Smile,
      title: "Mood Tracking",
      desc: "Visualize your emotional journey with daily logs and insightful analytics.",
      longDesc: "Understanding your emotional patterns is the first step toward better mental health. Our mood tracker allows you to log your feelings in seconds and provides beautiful, easy-to-read charts that show how your mood changes over weeks and months. Discover what triggers your best days and learn to navigate the difficult ones.",
      color: "text-blue-600",
      bg: "bg-blue-50",
      img: "https://images.unsplash.com/photo-1512438248247-f0f2a5a8b7f0?auto=format&fit=crop&w=800&q=80"
    },
    {
      icon: BookOpen,
      title: "Smart Journal",
      desc: "Write your thoughts and receive AI-powered insights and reflections.",
      longDesc: "Journaling is a powerful tool for self-reflection. Our Smart Journal goes beyond just storing your words; it uses AI to analyze your entries and provide gentle insights, identifying recurring themes and suggesting areas for growth. It's like having a conversation with your own thoughts, guided by a supportive friend.",
      color: "text-purple-600",
      bg: "bg-purple-50",
      img: "https://images.unsplash.com/photo-1517842645767-c639042777db?auto=format&fit=crop&w=800&q=80"
    },
    {
      icon: Wind,
      title: "Wellness Exercises",
      desc: "Guided breathing, meditation, and grounding techniques for instant calm.",
      longDesc: "When stress hits, you need tools that work immediately. We provide a library of guided exercises, including the 4-7-8 breathing technique, mindfulness meditation timers, and the 5-4-3-2-1 grounding method. These evidence-based practices help regulate your nervous system and bring you back to the present moment.",
      color: "text-orange-600",
      bg: "bg-orange-50",
      img: "https://images.unsplash.com/photo-1506126613408-eca07ce68773?auto=format&fit=crop&w=800&q=80"
    },
    {
      icon: AlertCircle,
      title: "Panic Support",
      desc: "Immediate assistance with calming guidance during overwhelming moments.",
      longDesc: "In moments of crisis, every second counts. Our Panic Button provides a dedicated, high-contrast interface designed to guide you through a panic attack. It offers rhythmic breathing animations, grounding affirmations, and step-by-step instructions to help you regain control and feel safe again.",
      color: "text-rose-600",
      bg: "bg-rose-50",
      img: "https://images.unsplash.com/photo-1499209974431-9dac3adaf471?auto=format&fit=crop&w=800&q=80"
    },
    {
      icon: Shield,
      title: "Private & Secure",
      desc: "Your data is encrypted and private. A safe space for your mental health.",
      longDesc: "We take your privacy with the utmost seriousness. All your conversations, journal entries, and mood logs are encrypted and stored securely. We never sell your data, and you have complete control over your information. This is your private sanctuary, and we are committed to keeping it that way.",
      color: "text-slate-600",
      bg: "bg-slate-50",
      img: "https://images.unsplash.com/photo-1563986768609-322da13575f3?auto=format&fit=crop&w=800&q=80"
    }
  ];

  return (
    <div className="min-h-screen bg-[#f7fcf9] relative overflow-hidden font-sans selection:bg-emerald-100 selection:text-emerald-900">
      {/* Watermark Background - Tree Forest */}
      <div className="fixed inset-0 pointer-events-none opacity-[0.04] z-0">
        <div 
          className="w-full h-full bg-repeat opacity-50"
          style={{ 
            backgroundImage: 'url("https://images.unsplash.com/photo-1441974231531-c6227db76b6e?auto=format&fit=crop&w=400&q=20")',
            backgroundSize: '400px'
          }}
        />
      </div>

      {/* Header */}
      <header className="relative z-10 flex items-center justify-between px-6 py-4 md:px-12 md:py-6 border-b border-emerald-100/50 bg-white/70 backdrop-blur-xl sticky top-0">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-emerald-600 rounded-xl flex items-center justify-center shadow-lg shadow-emerald-200">
            <Heart className="text-white" size={24} fill="currentColor" />
          </div>
          <span className="font-bold text-2xl text-slate-800 tracking-tight">AI Therapy Companion</span>
        </div>
        <div className="flex items-center gap-4">
          <button className="hidden sm:flex items-center gap-2 px-4 py-2 text-slate-600 font-bold hover:text-emerald-600 transition-colors">
            <HelpCircle size={20} /> Help
          </button>
          <button 
            onClick={onStart}
            disabled={isSigningIn}
            className={`px-6 py-2.5 bg-emerald-600 hover:bg-emerald-700 text-white rounded-full font-bold transition-all shadow-lg shadow-emerald-100 flex items-center gap-2 ${isSigningIn ? 'opacity-70 cursor-not-allowed' : ''}`}
          >
            {isSigningIn ? (
              <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
            ) : (
              <>Login <ChevronRight size={18} /></>
            )}
          </button>
        </div>
      </header>

      {/* Hero Section */}
      <section className="relative z-10 pt-24 pb-16 px-6 md:px-12 max-w-7xl mx-auto">
        <div className="grid lg:grid-cols-2 gap-16 items-center">
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8 }}
          >
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-emerald-100 text-emerald-700 rounded-full text-sm font-bold mb-8">
              <Sparkles size={16} /> Your 24/7 Mental Wellness Partner
            </div>
            <h1 className="text-5xl md:text-7xl font-bold text-slate-900 mb-8 leading-[1.1]">
              Find Peace in the <br />
              <span className="text-emerald-600">Digital Forest</span>
            </h1>
            <p className="text-xl text-slate-600 mb-10 leading-relaxed max-w-xl">
              A compassionate, non-judgmental space designed to support your mental journey. We combine AI empathy with proven therapeutic frameworks to help you thrive.
            </p>
            <div className="flex flex-col sm:flex-row items-center gap-4">
              <button 
                onClick={onStart}
                disabled={isSigningIn}
                className={`w-full sm:w-auto px-10 py-5 bg-emerald-600 hover:bg-emerald-700 text-white rounded-2xl font-bold text-lg transition-all shadow-xl shadow-emerald-200 hover:scale-[1.02] active:scale-95 flex items-center justify-center gap-3 ${isSigningIn ? 'opacity-70 cursor-not-allowed' : ''}`}
              >
                {isSigningIn ? (
                  <>
                    <div className="w-6 h-6 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    Signing in...
                  </>
                ) : (
                  'Start Your Journey'
                )}
              </button>
              <button className="w-full sm:w-auto px-10 py-5 bg-white border border-emerald-100 text-emerald-700 rounded-2xl font-bold text-lg hover:bg-emerald-50 transition-all">
                View Demo
              </button>
            </div>
            <div className="mt-12 flex items-center gap-6 text-slate-400">
              <div className="flex -space-x-3">
                {[1,2,3,4].map(i => (
                  <img key={i} src={`https://i.pravatar.cc/100?img=${i+10}`} className="w-10 h-10 rounded-full border-2 border-white" alt="user" />
                ))}
              </div>
              <p className="text-sm font-medium"><span className="text-slate-900 font-bold">10,000+</span> people finding calm today</p>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="relative"
          >
            <div className="absolute -inset-4 bg-emerald-200/30 rounded-[3rem] blur-3xl" />
            <img 
              src="https://images.unsplash.com/photo-1544367567-0f2fcb009e0b?auto=format&fit=crop&w=1200&q=80" 
              className="relative z-10 rounded-[3rem] shadow-2xl border-8 border-white"
              alt="Meditation"
            />
            <div className="absolute -bottom-6 -left-6 z-20 bg-white p-6 rounded-3xl shadow-xl border border-emerald-50 max-w-[240px]">
              <div className="flex items-center gap-3 mb-3">
                <div className="w-8 h-8 bg-emerald-100 rounded-full flex items-center justify-center text-emerald-600">
                  <Smile size={18} />
                </div>
                <p className="font-bold text-slate-800 text-sm">Mood Improved</p>
              </div>
              <p className="text-xs text-slate-500 leading-relaxed">"I feel much more centered after our chat today. Thank you for listening."</p>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Why Choose Section */}
      <section className="relative z-10 py-24 px-6 md:px-12 bg-white/50 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto">
          <div className="grid lg:grid-cols-3 gap-12">
            <div className="lg:col-span-1">
              <h2 className="text-4xl font-bold text-slate-900 mb-6 leading-tight">Why Choose <br /><span className="text-emerald-600">AI Therapy?</span></h2>
              <p className="text-slate-600 mb-8 leading-relaxed">
                Traditional therapy is invaluable, but it's not always accessible. We bridge the gap by providing immediate, affordable, and high-quality support whenever you need it.
              </p>
              <div className="flex items-center gap-4 p-4 bg-emerald-50 rounded-2xl border border-emerald-100">
                <Shield className="text-emerald-600" size={32} />
                <div>
                  <p className="font-bold text-slate-800">100% Private</p>
                  <p className="text-xs text-slate-500">Your data never leaves our secure forest.</p>
                </div>
              </div>
            </div>
            <div className="lg:col-span-2 grid sm:grid-cols-2 gap-8">
              {[
                { title: "Zero Judgment", desc: "Speak your truth without fear. Our AI is designed to be a neutral, supportive listener.", icon: Heart },
                { title: "24/7 Availability", desc: "Mental health doesn't follow a schedule. We're here for you at 3 AM or mid-afternoon.", icon: Zap },
                { title: "Affordable Care", desc: "Quality mental health support shouldn't be a luxury. Our companion is free to use.", icon: Sparkles },
                { title: "Evidence Based", desc: "Built on Cognitive Behavioral Therapy (CBT) principles used by professionals worldwide.", icon: CheckCircle2 }
              ].map((item, i) => (
                <div key={i} className="p-8 bg-white rounded-[2.5rem] border border-emerald-50 shadow-sm hover:shadow-md transition-all">
                  <div className="w-12 h-12 bg-emerald-50 text-emerald-600 rounded-xl flex items-center justify-center mb-6">
                    <item.icon size={24} />
                  </div>
                  <h4 className="text-xl font-bold text-slate-800 mb-3">{item.title}</h4>
                  <p className="text-slate-500 text-sm leading-relaxed">{item.desc}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Features Grid */}
      <section className="relative z-10 py-24 px-6 md:px-12 max-w-7xl mx-auto">
        <div className="text-center mb-20">
          <h2 className="text-4xl md:text-5xl font-bold text-slate-900 mb-6">Designed for Your Well-being</h2>
          <p className="text-slate-500 max-w-2xl mx-auto text-lg">Click on any feature to learn how it can help you navigate your mental health journey.</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: idx * 0.1 }}
              onClick={() => setSelectedFeature(feature)}
              className="bg-white p-10 rounded-[3rem] border border-emerald-50 shadow-sm hover:shadow-2xl hover:-translate-y-2 transition-all group cursor-pointer"
            >
              <div className={`w-16 h-16 ${feature.bg} ${feature.color} rounded-2xl flex items-center justify-center mb-8 group-hover:scale-110 transition-transform shadow-sm`}>
                <feature.icon size={32} />
              </div>
              <h3 className="text-2xl font-bold text-slate-800 mb-4">{feature.title}</h3>
              <p className="text-slate-500 leading-relaxed mb-6">{feature.desc}</p>
              <div className="flex items-center text-emerald-600 font-bold text-sm group-hover:gap-2 transition-all">
                Learn More <ChevronRight size={16} />
              </div>
            </motion.div>
          ))}
        </div>
      </section>

      {/* How it Works */}
      <section className="relative z-10 py-24 bg-emerald-900 text-white overflow-hidden">
        <div className="absolute top-0 right-0 w-1/2 h-full bg-emerald-800/50 skew-x-12 translate-x-1/4" />
        <div className="max-w-7xl mx-auto px-6 md:px-12 relative z-10">
          <div className="grid lg:grid-cols-2 gap-20 items-center">
            <div>
              <h2 className="text-4xl md:text-5xl font-bold mb-8 leading-tight">Science-Backed <br />Support System</h2>
              <div className="space-y-8">
                {[
                  { icon: CheckCircle2, title: "CBT Framework", desc: "We use evidence-based Cognitive Behavioral Therapy techniques to help reframe negative thoughts." },
                  { icon: Users, title: "Empathetic AI", desc: "Our models are fine-tuned to recognize emotional cues and respond with genuine compassion." },
                  { icon: Zap, title: "Instant Access", desc: "No waiting lists or appointments. Get the support you need the moment you need it." }
                ].map((item, i) => (
                  <div key={i} className="flex gap-6">
                    <div className="w-12 h-12 bg-emerald-500/20 rounded-2xl flex items-center justify-center flex-shrink-0">
                      <item.icon className="text-emerald-400" size={24} />
                    </div>
                    <div>
                      <h4 className="text-xl font-bold mb-2">{item.title}</h4>
                      <p className="text-emerald-100/70 leading-relaxed">{item.desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
            <div className="relative">
              <img 
                src="https://images.unsplash.com/photo-1516302752625-fcc3c50ae61f?auto=format&fit=crop&w=1000&q=80" 
                className="rounded-[3rem] shadow-2xl"
                alt="Nature"
              />
              <div className="absolute inset-0 bg-emerald-900/20 rounded-[3rem]" />
            </div>
          </div>
        </div>
      </section>

      {/* Testimonials */}
      <section className="relative z-10 py-24 px-6 md:px-12 max-w-7xl mx-auto">
        <div className="text-center mb-16">
          <h2 className="text-4xl font-bold text-slate-900 mb-4">Real Stories of Calm</h2>
          <p className="text-slate-500">Join thousands who have found their peace with us.</p>
        </div>
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          {[
            { name: "Sarah J.", text: "The AI therapist helped me through a major career transition. It felt like talking to a wise friend who never gets tired of listening." },
            { name: "Michael R.", text: "The panic button is a lifesaver. Having that guided breathing during an attack makes all the difference." },
            { name: "Elena K.", text: "I love the smart journal. The insights it gives me about my own patterns have been eye-opening for my personal growth." }
          ].map((t, i) => (
            <div key={i} className="bg-white p-8 rounded-3xl border border-emerald-50 shadow-sm italic text-slate-600 leading-relaxed">
              "{t.text}"
              <p className="mt-6 font-bold text-slate-900 not-italic">— {t.name}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Our Mission Section */}
      <section className="relative z-10 py-24 px-6 md:px-12 bg-emerald-50/50 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto">
          <div className="flex flex-col md:flex-row items-center gap-16">
            <div className="flex-1">
              <div className="w-16 h-16 bg-emerald-600 rounded-2xl flex items-center justify-center text-white mb-8 shadow-lg shadow-emerald-200">
                <Heart size={32} fill="currentColor" />
              </div>
              <h2 className="text-4xl md:text-5xl font-bold text-slate-900 mb-8 leading-tight">Our Mission: <br /><span className="text-emerald-600">Democratizing Mental Wellness</span></h2>
              <p className="text-xl text-slate-600 leading-relaxed mb-8">
                We believe that mental health support is a fundamental human right, not a privilege. Our mission is to provide every person on the planet with a compassionate, intelligent, and always-available companion to help them navigate the complexities of the human experience.
              </p>
              <div className="grid grid-cols-2 gap-6">
                <div>
                  <h4 className="font-bold text-slate-900 text-lg mb-2">Accessible</h4>
                  <p className="text-slate-500 text-sm">Available to anyone with an internet connection, anywhere in the world.</p>
                </div>
                <div>
                  <h4 className="font-bold text-slate-900 text-lg mb-2">Compassionate</h4>
                  <p className="text-slate-500 text-sm">Designed with empathy at its core to provide genuine emotional support.</p>
                </div>
              </div>
            </div>
            <div className="flex-1 relative">
              <div className="absolute -inset-4 bg-emerald-200/20 rounded-full blur-3xl animate-pulse" />
              <img 
                src="https://images.unsplash.com/photo-1518531933037-91b2f5f229cc?auto=format&fit=crop&w=800&q=80" 
                className="relative z-10 rounded-[3rem] shadow-2xl border-4 border-white transform rotate-3"
                alt="Nature and Growth"
              />
            </div>
          </div>
        </div>
      </section>

      {/* FAQ Section */}
      <section className="relative z-10 py-24 px-6 md:px-12 max-w-4xl mx-auto">
        <div className="text-center mb-16">
          <h2 className="text-4xl font-bold text-slate-900 mb-4">Frequently Asked Questions</h2>
          <p className="text-slate-500">Everything you need to know about your companion.</p>
        </div>
        <div className="space-y-4">
          {[
            { q: "Is this a replacement for professional therapy?", a: "No. AI Therapy Companion is a supportive tool designed to complement professional care or provide immediate support when a therapist is unavailable. If you are in crisis, please contact emergency services." },
            { q: "How secure is my data?", a: "Extremely. We use industry-standard encryption and never sell your personal data. Your conversations and journal entries are private to you." },
            { q: "What therapeutic techniques are used?", a: "Our AI is trained on Cognitive Behavioral Therapy (CBT) frameworks, mindfulness practices, and grounding techniques used by mental health professionals." },
            { q: "Is it really free?", a: "Yes. Our mission is to make mental health support accessible to everyone. All core features are completely free to use." }
          ].map((faq, i) => (
            <details key={i} className="group bg-white rounded-3xl border border-emerald-50 shadow-sm overflow-hidden transition-all">
              <summary className="flex items-center justify-between p-6 cursor-pointer font-bold text-slate-800 list-none">
                {faq.q}
                <ChevronRight className="group-open:rotate-90 transition-transform text-emerald-600" size={20} />
              </summary>
              <div className="px-6 pb-6 text-slate-600 leading-relaxed">
                {faq.a}
              </div>
            </details>
          ))}
        </div>
      </section>

      {/* Final CTA */}
      <section className="relative z-10 py-24 px-6 md:px-12 max-w-7xl mx-auto text-center">
        <div className="bg-emerald-600 rounded-[4rem] p-12 md:p-24 shadow-2xl shadow-emerald-200 relative overflow-hidden">
          <div className="absolute top-0 left-0 w-full h-full opacity-10 pointer-events-none">
            <div className="absolute top-[-20%] left-[-10%] w-[60%] h-[60%] bg-white rounded-full blur-[120px]" />
            <div className="absolute bottom-[-20%] right-[-10%] w-[60%] h-[60%] bg-white rounded-full blur-[120px]" />
          </div>
          <h2 className="text-4xl md:text-6xl font-bold text-white mb-8 relative z-10">Ready to find your calm?</h2>
          <p className="text-emerald-50 text-xl mb-12 max-w-2xl mx-auto relative z-10">
            Join thousands of others who are taking control of their mental well-being with AI Therapy Companion.
          </p>
          <button 
            onClick={onStart}
            disabled={isSigningIn}
            className="relative z-10 px-12 py-6 bg-white text-emerald-600 rounded-2xl font-bold text-xl hover:bg-emerald-50 transition-all shadow-xl hover:scale-105 active:scale-95"
          >
            Get Started for Free
          </button>
        </div>
      </section>

      {/* Footer */}
      <footer className="relative z-10 py-16 px-6 md:px-12 border-t border-emerald-50 bg-white">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-center gap-8">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-emerald-600 rounded-lg flex items-center justify-center">
              <Heart className="text-white" size={18} fill="currentColor" />
            </div>
            <span className="font-bold text-xl text-slate-800">AI Therapy Companion</span>
          </div>
          <div className="flex gap-8 text-sm font-bold text-slate-500">
            <a href="#" className="hover:text-emerald-600 transition-colors">Privacy</a>
            <a href="#" className="hover:text-emerald-600 transition-colors">Terms</a>
            <a href="#" className="hover:text-emerald-600 transition-colors">Contact</a>
          </div>
          <p className="text-slate-400 text-sm">© 2026 AI Therapy Companion. Created by <span className="font-bold text-emerald-600">Shyam Sundhr Karne</span>. All rights reserved.</p>
        </div>
      </footer>

      {/* Feature Modal */}
      <AnimatePresence>
        {selectedFeature && (
          <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 sm:p-6">
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setSelectedFeature(null)}
              className="absolute inset-0 bg-slate-900/60 backdrop-blur-md"
            />
            <motion.div
              initial={{ opacity: 0, scale: 0.9, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.9, y: 20 }}
              className="relative w-full max-w-2xl bg-white rounded-[3rem] shadow-2xl overflow-hidden"
            >
              <button 
                onClick={() => setSelectedFeature(null)}
                className="absolute top-6 right-6 p-2 text-slate-400 hover:text-slate-600 bg-slate-50 rounded-full z-10"
              >
                <X size={24} />
              </button>
              
              <div className="grid md:grid-cols-2">
                <div className="h-64 md:h-full">
                  <img src={selectedFeature.img} className="w-full h-full object-cover" alt={selectedFeature.title} />
                </div>
                <div className="p-8 md:p-12">
                  <div className={`w-12 h-12 ${selectedFeature.bg} ${selectedFeature.color} rounded-xl flex items-center justify-center mb-6`}>
                    <selectedFeature.icon size={24} />
                  </div>
                  <h2 className="text-3xl font-bold text-slate-900 mb-4">{selectedFeature.title}</h2>
                  <p className="text-slate-600 leading-relaxed mb-8">
                    {selectedFeature.longDesc}
                  </p>
                  <button 
                    onClick={() => {
                      if (!isSigningIn) {
                        setSelectedFeature(null);
                        onStart();
                      }
                    }}
                    disabled={isSigningIn}
                    className={`w-full py-4 bg-emerald-600 text-white rounded-2xl font-bold hover:bg-emerald-700 transition-all shadow-lg shadow-emerald-100 flex items-center justify-center gap-3 ${isSigningIn ? 'opacity-70 cursor-not-allowed' : ''}`}
                  >
                    {isSigningIn ? (
                      <>
                        <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                        Signing in...
                      </>
                    ) : (
                      'Try it Now'
                    )}
                  </button>
                </div>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default LandingPage;
