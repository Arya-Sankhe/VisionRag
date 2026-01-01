// Dynamic API URL based on browser location (works for both localhost and VPS)
const getApiBase = () => {
    if (typeof window !== 'undefined') {
        // In browser: use same hostname but port 8000
        return `http://${window.location.hostname}:8000`;
    }
    // During SSR: use environment variable or default
    return process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
};

const API_BASE = getApiBase();

// Types
export type ModelMode = 'fast' | 'deep';

export interface DocumentInfo {
    id: string;
    name: string;
    page_count: number;
}

export interface RetrievedPage {
    doc_id: number;
    page_num: number;
    score: number;
    image_base64?: string;
}

export interface ChatResponse {
    answer: string;
    sources: RetrievedPage[];
    mode: ModelMode;
}

export interface UploadResponse {
    success: boolean;
    message: string;
    documents_added: number;
    indexing_status: {
        fast: Array<{ file: string; status: string; pages?: number }>;
        deep: Array<{ file: string; status: string; pages?: number }>;
    };
}

// API Functions
export async function uploadDocuments(files: File[]): Promise<UploadResponse> {
    const formData = new FormData();
    files.forEach(file => formData.append('files', file));

    const response = await fetch(`${API_BASE}/api/v1/documents/upload`, {
        method: 'POST',
        body: formData,
    });

    return response.json();
}

export async function getDocuments(mode: ModelMode = 'fast'): Promise<{ documents: DocumentInfo[]; total: number }> {
    const response = await fetch(`${API_BASE}/api/v1/documents?mode=${mode}`);
    return response.json();
}

export async function clearDocuments(): Promise<{ success: boolean }> {
    const response = await fetch(`${API_BASE}/api/v1/documents/clear`, {
        method: 'DELETE',
    });
    return response.json();
}

export async function sendMessage(query: string, mode: ModelMode = 'fast'): Promise<ChatResponse> {
    const response = await fetch(`${API_BASE}/api/v1/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, mode, include_images: true }),
    });
    return response.json();
}
