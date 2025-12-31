'use client';

import ReactMarkdown from 'react-markdown';
import { RetrievedPage } from '@/lib/api';

interface Message {
    role: 'user' | 'assistant';
    content: string;
    sources?: RetrievedPage[];
}

export function MessageBubble({ message }: { message: Message }) {
    const isUser = message.role === 'user';

    return (
        <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
            <div
                className={`max-w-[80%] rounded-lg px-4 py-3 ${isUser
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-700 text-gray-100'
                    }`}
            >
                {/* Message content */}
                <div className="prose prose-invert prose-sm max-w-none">
                    <ReactMarkdown>{message.content}</ReactMarkdown>
                </div>

                {/* Source images for assistant messages */}
                {!isUser && message.sources && message.sources.length > 0 && (
                    <div className="mt-4 pt-3 border-t border-gray-600">
                        <p className="text-xs text-gray-400 mb-2">📄 Retrieved Pages:</p>
                        <div className="flex gap-2 overflow-x-auto pb-2">
                            {message.sources.slice(0, 3).map((source, i) => (
                                <div key={i} className="flex-shrink-0">
                                    {source.image_base64 && (
                                        <img
                                            src={`data:image/png;base64,${source.image_base64}`}
                                            alt={`Page ${source.page_num}`}
                                            className="h-32 rounded border border-gray-500 cursor-pointer hover:opacity-80"
                                            onClick={() => {
                                                // Open full image in new tab
                                                const w = window.open();
                                                if (w) {
                                                    w.document.write(`<img src="data:image/png;base64,${source.image_base64}" />`);
                                                }
                                            }}
                                        />
                                    )}
                                    <p className="text-xs text-gray-400 mt-1 text-center">
                                        Page {source.page_num} ({(source.score * 100).toFixed(0)}%)
                                    </p>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
