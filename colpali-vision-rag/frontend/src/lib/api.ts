const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

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
}

export async function uploadDocuments(files: File[]): Promise<{ success: boolean; message: string }> {
    const formData = new FormData();
    files.forEach(file => formData.append('files', file));

    const response = await fetch(`${API_BASE}/api/v1/documents/upload`, {
        method: 'POST',
        body: formData,
    });

    return response.json();
}

export async function getDocuments(): Promise<{ documents: DocumentInfo[]; total: number }> {
    const response = await fetch(`${API_BASE}/api/v1/documents`);
    return response.json();
}

export async function clearDocuments(): Promise<{ success: boolean }> {
    const response = await fetch(`${API_BASE}/api/v1/documents/clear`, {
        method: 'DELETE',
    });
    return response.json();
}

export async function sendMessage(query: string): Promise<ChatResponse> {
    const response = await fetch(`${API_BASE}/api/v1/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, include_images: true }),
    });
    return response.json();
}
