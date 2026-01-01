'use client';

import { useState } from 'react';
import { sendMessage, RetrievedPage, ModelMode } from '@/lib/api';
import { MessageBubble } from './MessageBubble';

interface Message {
    role: 'user' | 'assistant';
    content: string;
    sources?: RetrievedPage[];
    mode?: ModelMode;
}

export function ChatInterface() {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [mode, setMode] = useState<ModelMode>('fast');

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!input.trim() || isLoading) return;

        const userMessage = input.trim();
        setInput('');
        setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
        setIsLoading(true);

        try {
            const response = await sendMessage(userMessage, mode);
            setMessages(prev => [
                ...prev,
                {
                    role: 'assistant',
                    content: response.answer,
                    sources: response.sources,
                    mode: response.mode,
                },
            ]);
        } catch (error) {
            setMessages(prev => [
                ...prev,
                { role: 'assistant', content: '❌ Error: Failed to get response' },
            ]);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="flex flex-col h-full">
            {/* Mode Toggle */}
            <div className="p-4 border-b border-gray-700 flex items-center justify-between">
                <span className="text-sm text-gray-400">Retrieval Mode:</span>
                <div className="flex bg-gray-800 rounded-lg p-1">
                    <button
                        onClick={() => setMode('fast')}
                        className={`px-4 py-1.5 rounded-md text-sm font-medium transition-all ${mode === 'fast'
                                ? 'bg-green-600 text-white shadow-sm'
                                : 'text-gray-400 hover:text-white'
                            }`}
                    >
                        ⚡ Fast
                    </button>
                    <button
                        onClick={() => setMode('deep')}
                        className={`px-4 py-1.5 rounded-md text-sm font-medium transition-all ${mode === 'deep'
                                ? 'bg-purple-600 text-white shadow-sm'
                                : 'text-gray-400 hover:text-white'
                            }`}
                    >
                        🔍 Deep
                    </button>
                </div>
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {messages.length === 0 && (
                    <div className="text-center text-gray-400 mt-20">
                        <p className="text-lg">👋 Welcome to ColPali Vision RAG</p>
                        <p className="text-sm mt-2">Upload documents and ask questions about them</p>
                        <div className="mt-4 text-xs">
                            <p className="text-green-400">⚡ Fast: ColSmol-500M - Quick responses</p>
                            <p className="text-purple-400 mt-1">🔍 Deep: ColPali-v1.3 - More accurate</p>
                        </div>
                    </div>
                )}
                {messages.map((msg, i) => (
                    <MessageBubble key={i} message={msg} />
                ))}
                {isLoading && (
                    <div className="flex justify-start">
                        <div className="bg-gray-700 rounded-lg px-4 py-2 animate-pulse">
                            {mode === 'fast' ? '⚡ Fast' : '🔍 Deep'} thinking...
                        </div>
                    </div>
                )}
            </div>

            {/* Input */}
            <form onSubmit={handleSubmit} className="p-4 border-t border-gray-700">
                <div className="flex gap-2">
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder="Ask about your documents..."
                        className="flex-1 bg-gray-800 border border-gray-600 rounded-lg px-4 py-2 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
                        disabled={isLoading}
                    />
                    <button
                        type="submit"
                        disabled={isLoading || !input.trim()}
                        className={`px-6 py-2 rounded-lg font-medium transition-colors ${mode === 'fast'
                                ? 'bg-green-600 hover:bg-green-700 disabled:bg-gray-600'
                                : 'bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600'
                            }`}
                    >
                        Send
                    </button>
                </div>
            </form>
        </div>
    );
}
