import React from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { AlertTriangle, X } from 'lucide-react';

interface ConfirmModalProps {
  isOpen: boolean;
  title: string;
  message: string;
  confirmLabel?: string;
  cancelLabel?: string;
  onConfirm: () => void;
  onCancel: () => void;
  variant?: 'danger' | 'warning' | 'info';
}

const ConfirmModal: React.FC<ConfirmModalProps> = ({
  isOpen,
  title,
  message,
  confirmLabel = 'Confirm',
  cancelLabel = 'Cancel',
  onConfirm,
  onCancel,
  variant = 'danger'
}) => {
  return (
    <AnimatePresence>
      {isOpen && (
        <div className="fixed inset-0 z-[200] flex items-center justify-center p-4">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onCancel}
            className="absolute inset-0 bg-slate-900/60 backdrop-blur-sm"
          />
          <motion.div
            initial={{ opacity: 0, scale: 0.9, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9, y: 20 }}
            className="relative w-full max-w-md bg-white rounded-[2.5rem] shadow-2xl overflow-hidden p-8 text-center"
          >
            <div className={`w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-6 ${
              variant === 'danger' ? 'bg-rose-50 text-rose-500' : 
              variant === 'warning' ? 'bg-orange-50 text-orange-500' : 
              'bg-blue-50 text-blue-500'
            }`}>
              <AlertTriangle size={32} />
            </div>
            
            <h2 className="text-2xl font-bold text-slate-900 mb-3">{title}</h2>
            <p className="text-slate-500 mb-8 leading-relaxed">
              {message}
            </p>
            
            <div className="flex flex-col sm:flex-row gap-3">
              <button
                onClick={onCancel}
                className="flex-1 py-4 bg-slate-100 text-slate-600 rounded-2xl font-bold hover:bg-slate-200 transition-all"
              >
                {cancelLabel}
              </button>
              <button
                onClick={onConfirm}
                className={`flex-1 py-4 text-white rounded-2xl font-bold transition-all shadow-lg ${
                  variant === 'danger' ? 'bg-rose-600 hover:bg-rose-700 shadow-rose-100' : 
                  variant === 'warning' ? 'bg-orange-600 hover:bg-orange-700 shadow-orange-100' : 
                  'bg-blue-600 hover:bg-blue-700 shadow-blue-100'
                }`}
              >
                {confirmLabel}
              </button>
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
};

export default ConfirmModal;
