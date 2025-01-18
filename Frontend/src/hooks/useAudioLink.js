// src/hooks/useAudioLink.js
import { useState } from 'react';

export function useAudioLink() {
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);

  const processLink = async (url) => {
    console.log('Processing URL:', url);
    setIsProcessing(true);
    setError(null);

    try {
      // First validate URL format
      new URL(url); // Will throw if invalid URL

      console.log('Making fetch request to backend...');
      const response = await fetch('http://localhost:5001/process-audio-url', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include', // Add this for CORS
        body: JSON.stringify({ url })
      });

      console.log('Response status:', response.status);
      console.log('Response headers:', Object.fromEntries(response.headers.entries()));

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: 'Network error' }));
        console.error('Error response:', errorData);
        throw new Error(errorData.error || `Failed to process URL (${response.status})`);
      }

      // Get the audio data as a blob
      const audioBlob = await response.blob();
      console.log('Received blob:', audioBlob);
      console.log('Blob size:', audioBlob.size);
      console.log('Blob type:', audioBlob.type);
      
      if (audioBlob.size === 0) {
        throw new Error('Received empty audio data');
      }

      // Create a File object that can be used with the existing handleFileChange
      const audioFile = new File([audioBlob], 'youtube-audio.wav', {
        type: 'audio/wav',
        lastModified: new Date().getTime()
      });

      console.log('Created audio file:', audioFile);
      console.log('File size:', audioFile.size);
      console.log('File type:', audioFile.type);

      return { audioFile };

    } catch (err) {
      console.error('Error in processLink:', err);
      if (err instanceof TypeError && err.message.includes('URL')) {
        const errorMsg = 'Please enter a valid URL';
        setError(errorMsg);
        throw new Error(errorMsg);
      } else {
        const errorMsg = err.message || 'Failed to process URL';
        setError(errorMsg);
        throw new Error(errorMsg);
      }
    } finally {
      console.log('Processing completed');
      setIsProcessing(false);
    }
  };

  const clearError = () => {
    setError(null);
  };

  return {
    isProcessing,
    error,
    processLink,
    clearError
  };
}