import { ChatInterface } from '@/components/ChatInterface';
import { DocumentManager } from '@/components/DocumentManager';

export default function Home() {
    return (
        <main className="flex h-screen bg-gray-900 text-white">
            {/* Sidebar */}
            <aside className="w-80 border-r border-gray-700 overflow-y-auto">
                <div className="p-4 border-b border-gray-700">
                    <h1 className="text-xl font-bold">🔍 ColPali Vision RAG</h1>
                    <p className="text-xs text-gray-400 mt-1">Visual document retrieval</p>
                </div>
                <DocumentManager />
            </aside>

            {/* Main chat area */}
            <section className="flex-1 flex flex-col">
                <ChatInterface />
            </section>
        </main>
    );
}
