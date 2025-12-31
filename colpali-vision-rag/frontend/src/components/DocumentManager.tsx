'use client';

import { useCallback, useEffect, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { uploadDocuments, getDocuments, clearDocuments, DocumentInfo } from '@/lib/api';

export function DocumentManager() {
    const [documents, setDocuments] = useState<DocumentInfo[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [isUploading, setIsUploading] = useState(false);
    const [message, setMessage] = useState<string | null>(null);
    const [selectedFiles, setSelectedFiles] = useState<File[]>([]);

    const fetchDocuments = useCallback(async () => {
        setIsLoading(true);
        try {
            const response = await getDocuments();
            setDocuments(response.documents);
        } catch (error) {
            console.error('Failed to fetch documents:', error);
        } finally {
            setIsLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchDocuments();
    }, [fetchDocuments]);

    const onDrop = useCallback((acceptedFiles: File[]) => {
        setSelectedFiles(prev => [...prev, ...acceptedFiles]);
        setMessage(null);
    }, []);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: { 'application/pdf': ['.pdf'] },
        multiple: true,
    });

    const handleUpload = async () => {
        if (selectedFiles.length === 0) return;
        setIsUploading(true);
        setMessage(null);

        try {
            const result = await uploadDocuments(selectedFiles);
            setMessage(result.message);
            setSelectedFiles([]);
            await fetchDocuments();
        } catch (error) {
            setMessage('❌ Upload failed');
        } finally {
            setIsUploading(false);
        }
    };

    const handleClear = async () => {
        if (!confirm('Clear all documents?')) return;
        setIsLoading(true);
        try {
            await clearDocuments();
            setDocuments([]);
            setMessage('🗑️ All documents cleared');
        } catch (error) {
            setMessage('❌ Clear failed');
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="p-4 space-y-4">
            <h2 className="text-lg font-semibold">📚 Documents</h2>

            {/* Dropzone */}
            <div
                {...getRootProps()}
                className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors ${isDragActive ? 'border-blue-500 bg-blue-500/10' : 'border-gray-600 hover:border-gray-400'
                    }`}
            >
                <input {...getInputProps()} />
                <p className="text-gray-400">Drop PDFs here or click to select</p>
            </div>

            {/* Selected files */}
            {selectedFiles.length > 0 && (
                <div className="space-y-2">
                    <p className="text-sm text-gray-400">Selected: {selectedFiles.length} file(s)</p>
                    <div className="flex flex-wrap gap-2">
                        {selectedFiles.map((f, i) => (
                            <span
                                key={i}
                                className="bg-gray-700 px-2 py-1 rounded text-sm cursor-pointer hover:bg-red-600"
                                onClick={() => setSelectedFiles(prev => prev.filter((_, idx) => idx !== i))}
                            >
                                {f.name} ✕
                            </span>
                        ))}
                    </div>
                    <button
                        onClick={handleUpload}
                        disabled={isUploading}
                        className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 py-2 rounded-lg font-medium"
                    >
                        {isUploading ? 'Uploading...' : 'Upload Documents'}
                    </button>
                </div>
            )}

            {message && <p className="text-sm text-gray-300">{message}</p>}

            {/* Document list */}
            <div className="space-y-2">
                <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">{documents.length} document(s) indexed</span>
                    <button
                        onClick={handleClear}
                        disabled={documents.length === 0 || isLoading}
                        className="text-xs text-red-400 hover:text-red-300 disabled:text-gray-600"
                    >
                        Clear All
                    </button>
                </div>
                {documents.map((doc) => (
                    <div key={doc.id} className="bg-gray-800 rounded px-3 py-2 text-sm flex justify-between">
                        <span>{doc.name}</span>
                        <span className="text-gray-500">{doc.page_count} pages</span>
                    </div>
                ))}
            </div>
        </div>
    );
}
