import { useState, useRef, useCallback, useEffect } from 'react';

export const useTranslation = () => {
  // State management for audio and UI
  const [audioStatus, setAudioStatus] = useState('idle');
  const [audioReady, setAudioReady] = useState(false);
  const [file, setFile] = useState(null);
  const [targetLanguage, setTargetLanguage] = useState('fra');
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState('');
  const [translatedAudioUrl, setTranslatedAudioUrl] = useState('');
  const [isPlaying, setIsPlaying] = useState(false);
  const [progressText, setProgressText] = useState('');
  const [sourceText, setSourceText] = useState('');
  const [targetText, setTargetText] = useState('');
  const [showTranscript, setShowTranscript] = useState(false);

  // Refs for managing audio, cleanup, and async operations
  const audioRef = useRef(null);
  const abortControllerRef = useRef(null);
  const progressIntervalRef = useRef(null);
  const blobUrlRef = useRef(null);

  // Progress message handler with informative status updates
  const getProgressMessage = (progress) => {
    if (progress < 20) return "Preparing your audio for translation...";
    if (progress < 40) return "Analyzing speech patterns...";
    if (progress < 60) return "Converting to target language...";
    if (progress < 80) return "Generating natural speech...";
    if (progress < 100) return "Finalizing your translation...";
    return "Translation complete!";
  };

  // Enhanced cleanup function with comprehensive resource management
  const cleanup = useCallback(() => {
    try {
      // Clear previous audio state
      if (audioRef.current) {
        const audio = audioRef.current;
        
        // Properly stop audio playback
        audio.pause();
        audio.currentTime = 0;
        
        // Remove event listeners to prevent memory leaks
        audio.onloadstart = null;
        audio.oncanplaythrough = null;
        audio.onplay = null;
        audio.onpause = null;
        audio.onended = null;
        audio.onerror = null;
        
        // Clear the source
        audio.src = '';
        audio.load();
      }

      // Clean up Blob URL
      if (blobUrlRef.current) {
        URL.revokeObjectURL(blobUrlRef.current);
        blobUrlRef.current = null;
      }

      // Abort any pending requests
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
        abortControllerRef.current = null;
      }

      // Clear progress interval
      if (progressIntervalRef.current) {
        clearInterval(progressIntervalRef.current);
        progressIntervalRef.current = null;
      }

      // Reset all state
      setAudioReady(false);
      setAudioStatus('idle');
      setError('');
      setProgress(0);
      setProgressText('');
      setIsPlaying(false);
      setTranslatedAudioUrl('');
      setSourceText('');
      setTargetText('');
      setShowTranscript(false);

    } catch (e) {
      console.error('Cleanup error:', e);
      setError(`Cleanup failed: ${e.message}`);
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      cleanup();
    };
  }, [cleanup]);

  // Enhanced file validation and handling
  const handleFileChange = (event) => {
    cleanup();
    setError('');
    
    const selectedFile = event.target.files[0];
    if (selectedFile) {
      console.log('Selected file:', selectedFile.name, 'Type:', selectedFile.type);
      
      const validExtensions = ['.mp3', '.wav', '.ogg', '.m4a'];
      const fileExtension = selectedFile.name.toLowerCase().slice(selectedFile.name.lastIndexOf('.'));
      
      const validMimeTypes = [
        'audio/mp3', 'audio/mpeg', 'audio/wav', 'audio/wave',
        'audio/x-wav', 'audio/ogg', 'audio/x-m4a', 'audio/mp4', 'audio/aac'
      ];

      if (!validExtensions.includes(fileExtension)) {
        setError(`Invalid file extension. Please upload a file with extension: ${validExtensions.join(', ')}`);
        setFile(null);
        return;
      }

      if (!validMimeTypes.includes(selectedFile.type) && selectedFile.type !== '') {
        console.warn('File MIME type:', selectedFile.type);
        console.warn('Valid MIME types:', validMimeTypes);
      }

      // Additional size validation
      const maxSize = 50 * 1024 * 1024; // 50MB
      if (selectedFile.size > maxSize) {
        setError('File size exceeds 50MB limit');
        setFile(null);
        return;
      }

      setFile(selectedFile);
    } else {
      setFile(null);
    }
  };

  // Language selection with cleanup
  const handleLanguageSelect = (value) => {
    cleanup();
    setTargetLanguage(value);
  };

  // Enhanced audio playback control with better error handling
  const handlePlayPause = async () => {
    if (!audioRef.current) {
      console.error('No audio element reference');
      return;
    }

    try {
      if (audioRef.current.paused) {
        setAudioStatus('loading');
        const playAttempt = audioRef.current.play();
        
        if (playAttempt !== undefined) {
          await playAttempt;
          setIsPlaying(true);
          setAudioStatus('playing');
        }
      } else {
        audioRef.current.pause();
        setIsPlaying(false);
        setAudioStatus('ready');
      }
    } catch (error) {
      console.error('Playback error:', {
        error,
        audioState: {
          currentTime: audioRef.current.currentTime,
          duration: audioRef.current.duration,
          readyState: audioRef.current.readyState,
          networkState: audioRef.current.networkState,
          src: audioRef.current.src || 'no source'
        }
      });
      setError(`Playback failed: ${error.message}`);
      setAudioStatus('error');
      setIsPlaying(false);
    }
  };

  // Enhanced translation process with robust error handling and progress tracking
  // Now accepts a backend parameter with 'seamless' as default
  const processAudio = async (backend = 'seamless') => {
    cleanup();
    abortControllerRef.current = new AbortController();

    try {
      setProcessing(true);
      setProgress(10);
      setProgressText(getProgressMessage(10));
      setAudioStatus('loading');

      // Set up progress simulation
      progressIntervalRef.current = setInterval(() => {
        setProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressIntervalRef.current);
            return prev;
          }
          const increment = Math.random() * 15;
          const newProgress = Math.min(prev + increment, 90);
          setProgressText(getProgressMessage(newProgress));
          return newProgress;
        });
      }, 2000);

      // Prepare and validate form data
      const formData = new FormData();
      if (!file) {
        throw new Error('No file selected');
      }
      formData.append('file', file);
      formData.append('target_language', targetLanguage);
      formData.append('backend', backend); // Add the backend parameter

      // Log which backend is being used
      console.log(`Using translation backend: ${backend}`);

      // Make request with timeout and abort controller
      const response = await fetch('http://localhost:5001/translate', {
        method: 'POST',
        body: formData,
        credentials: 'include',
        signal: abortControllerRef.current.signal,
        headers: {
          'Accept': 'application/json'
        }
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || `Server error: ${response.status}`);
      }

      // Process response
      const responseData = await response.json();
      
      // Validate audio data
      if (!responseData.audio) {
        throw new Error('No audio data received from server');
      }

      // Process audio data with enhanced error handling
      const audioBuffer = Uint8Array.from(atob(responseData.audio), c => c.charCodeAt(0));
      if (!audioBuffer || audioBuffer.length === 0) {
        throw new Error('Failed to decode audio data');
      }

      // Create and validate audio blob
      const audioBlob = new Blob([audioBuffer], { type: 'audio/wav' });
      if (audioBlob.size === 0) {
        throw new Error('Created audio blob is empty');
      }

      // Create URL first before any state updates
      const url = URL.createObjectURL(audioBlob);
      console.log('Created audio URL:', url);
      
      // Store URL reference for cleanup
      if (blobUrlRef.current) {
        URL.revokeObjectURL(blobUrlRef.current);
      }
      blobUrlRef.current = url;
      
      // Set URL state
      setTranslatedAudioUrl(url);

      // Set transcripts if available
      if (responseData.transcripts) {
        setSourceText(responseData.transcripts.source || '');
        setTargetText(responseData.transcripts.target || '');
      }
      
      // Update UI states
      setProgress(100);
      setProgressText(getProgressMessage(100));
      setAudioReady(true);
      setAudioStatus('ready');
      setShowTranscript(true);

    } catch (e) {
      console.error('Translation error:', e);
      setError(e.message);
      setAudioStatus('error');
      cleanup();
    } finally {
      setProcessing(false);
      if (progressIntervalRef.current) {
        clearInterval(progressIntervalRef.current);
        progressIntervalRef.current = null;
      }
    }
  };

  // Return all necessary state and handlers
  return {
    // State
    audioStatus,
    audioReady,
    file,
    targetLanguage,
    processing,
    progress,
    error,
    translatedAudioUrl,
    isPlaying,
    progressText,
    sourceText,
    targetText,
    showTranscript,
    
    // Refs
    audioRef,
    
    // Handlers
    handleFileChange,
    handleLanguageSelect,
    handlePlayPause,
    processAudio,
    cleanup,
    
    // Setters
    setAudioStatus,
    setAudioReady,
    setError,
    setIsPlaying,
    setShowTranscript
  };
};