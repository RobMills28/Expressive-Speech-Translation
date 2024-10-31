import React, { useState, useRef, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "./components/ui/card"
import { Button } from "./components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./components/ui/select"
import { Label } from "./components/ui/label"
import { Input } from "./components/ui/input"
import { AlertCircle, Globe, Mic, Play, Pause } from 'lucide-react'
import { Alert, AlertDescription, AlertTitle } from "./components/ui/alert"
import { Progress } from "./components/ui/progress"

const LinguaSyncApp = () => {
  const [file, setFile] = useState(null);
  const [targetLanguage, setTargetLanguage] = useState('');
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState('');
  const [translatedAudioUrl, setTranslatedAudioUrl] = useState('');
  const [isPlaying, setIsPlaying] = useState(false);
  const [progressText, setProgressText] = useState('');
  const audioRef = useRef(null);
  const abortControllerRef = useRef(null);

  // Cleanup audio URL on unmount
  useEffect(() => {
    return () => {
      if (translatedAudioUrl) {
        URL.revokeObjectURL(translatedAudioUrl);
      }
    };
  }, [translatedAudioUrl]);

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile) {
      console.log('Selected file:', selectedFile.name, 'Type:', selectedFile.type);
      
      const validExtensions = ['.mp3', '.wav', '.ogg', '.m4a'];
      const fileExtension = selectedFile.name.toLowerCase().slice(selectedFile.name.lastIndexOf('.'));
      
      const validMimeTypes = [
        'audio/mp3',
        'audio/mpeg',
        'audio/wav',
        'audio/wave',
        'audio/x-wav',
        'audio/ogg',
        'audio/x-m4a',
        'audio/mp4',
        'audio/aac'
      ];

      if (!validExtensions.includes(fileExtension)) {
        setError(`Invalid file extension. Please upload a file with extension: ${validExtensions.join(', ')}`);
        return;
      }

      if (!validMimeTypes.includes(selectedFile.type) && selectedFile.type !== '') {
        console.warn('File MIME type:', selectedFile.type);
        console.warn('Valid MIME types:', validMimeTypes);
      }

      // Cleanup previous audio URL if exists
      if (translatedAudioUrl) {
        URL.revokeObjectURL(translatedAudioUrl);
      }

      setFile(selectedFile);
      setError('');
      setTranslatedAudioUrl('');
      setProgress(0);
      setProgressText('');
    }
  };

  const handleLanguageChange = (value) => {
    console.log('Language selected:', value);
    setTargetLanguage(value);
    setError('');
    setTranslatedAudioUrl('');
    setProgress(0);
    setProgressText('');
  };

  const handlePlayPause = () => {
    if (audioRef.current) {
      if (audioRef.current.paused) {
        const playPromise = audioRef.current.play();
        if (playPromise) {
          playPromise.catch((error) => {
            console.error('Error playing audio:', error);
            setError('Error playing audio: ' + error.message);
          });
        }
        setIsPlaying(true);
      } else {
        audioRef.current.pause();
        setIsPlaying(false);
      }
    }
  };

  const processAudio = async () => {
    if (!file || !targetLanguage) {
      setError('Please select both an audio file and a target language.');
      return;
    }

    // Cancel any existing request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();

    // Cleanup previous audio URL if exists
    if (translatedAudioUrl) {
      URL.revokeObjectURL(translatedAudioUrl);
      setTranslatedAudioUrl('');
    }

    setProcessing(true);
    setProgress(0);
    setProgressText('Loading model and processing audio...');
    setError('');

    const formData = new FormData();
    formData.append('file', file);
    formData.append('target_language', targetLanguage);

    try {
      const response = await fetch('http://localhost:5001/translate', {
        method: 'POST',
        body: formData,
        signal: abortControllerRef.current.signal
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }

      // Check if the response is an EventSource
      if (response.headers.get('content-type')?.includes('text/event-stream')) {
        const reader = response.body.getReader();
        let audioData = [];

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          // Parse the chunk as JSON
          const text = new TextDecoder().decode(value);
          const lines = text.split('\n');
          
          for (const line of lines) {
            if (line.startsWith('data:')) {
              try {
                const data = JSON.parse(line.slice(5));
                if (data.progress !== undefined) {
                  setProgress(data.progress);
                  setProgressText(data.status || 'Processing...');
                } else if (data.audio) {
                  audioData.push(data.audio);
                }
              } catch (e) {
                console.error('Error parsing event data:', e);
              }
            }
          }
        }

        // Combine all audio chunks and create URL
        const blob = new Blob(audioData, { type: 'audio/wav' });
        const url = URL.createObjectURL(blob);
        setTranslatedAudioUrl(url);
        setProgress(100);
        setProgressText('Translation complete!');
      } else {
        // Handle regular response
        const blob = await response.blob();
        if (blob.size === 0) {
          throw new Error('Received empty response from server');
        }
        const url = URL.createObjectURL(blob);
        setTranslatedAudioUrl(url);
        setProgress(100);
        setProgressText('Translation complete!');
      }
    } catch (e) {
      if (e.name === 'AbortError') {
        setError('Translation cancelled');
      } else {
        console.error('Error details:', e);
        setError(`An error occurred: ${e.message}`);
      }
    } finally {
      setProcessing(false);
      abortControllerRef.current = null;
    }
  };

  return (
    <div className="flex items-center justify-center min-h-screen bg-gradient-to-br from-purple-700 via-fuchsia-500 to-pink-500">
      <Card className="w-[400px] bg-white/90 backdrop-blur-md shadow-xl">
        <CardHeader className="bg-gradient-to-r from-fuchsia-600 to-pink-600 text-white rounded-t-lg">
          <CardTitle className="text-3xl font-bold">LinguaSync AI</CardTitle>
          <CardDescription className="text-pink-100">Audio Translation</CardDescription>
        </CardHeader>
        <CardContent className="mt-4 space-y-6">
          <div className="bg-fuchsia-100 p-4 rounded-lg shadow-inner">
            <Label htmlFor="audio-input" className="text-sm font-medium text-fuchsia-800 flex items-center">
              <Mic className="mr-2" size={18} />
              Upload Audio
            </Label>
            <div className="relative mt-1">
              <Input 
                id="audio-input" 
                type="file" 
                accept=".mp3,.wav,.ogg,.m4a,audio/*" 
                onChange={handleFileChange} 
                className="hidden"
              />
              <label 
                htmlFor="audio-input" 
                className="cursor-pointer inline-flex items-center px-4 py-2 bg-white border border-fuchsia-300 rounded-md font-semibold text-xs text-fuchsia-700 uppercase tracking-widest shadow-sm hover:bg-fuchsia-50 focus:outline-none focus:border-fuchsia-300 focus:ring focus:ring-fuchsia-200 active:bg-fuchsia-100 disabled:opacity-25 transition"
              >
                Choose file
              </label>
              <span className="ml-3 text-sm text-fuchsia-600">
                {file ? file.name : "No file chosen"}
              </span>
            </div>
          </div>
          
          <div className="space-y-2">
            <Label className="text-sm font-medium text-fuchsia-800 flex items-center">
              <Globe className="mr-2" size={18} />
              Target Language
            </Label>
            <Select 
              value={targetLanguage} 
              onValueChange={handleLanguageChange}
            >
              <SelectTrigger className="w-full border-fuchsia-300">
                <SelectValue placeholder="Select language" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="deu">🇩🇪 German</SelectItem>
                <SelectItem value="fra">🇫🇷 French</SelectItem>
                <SelectItem value="spa">🇪🇸 Spanish</SelectItem>
                <SelectItem value="ita">🇮🇹 Italian</SelectItem>
                <SelectItem value="por">🇵🇹 Portuguese</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {processing && (
            <div className="space-y-2">
              <Progress value={progress} className="w-full" />
              <p className="text-center text-sm text-fuchsia-800">{progressText}</p>
            </div>
          )}

          {error && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>Error</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {translatedAudioUrl && (
            <div className="mt-4 space-y-2">
              <audio 
                ref={audioRef} 
                src={translatedAudioUrl} 
                onEnded={() => setIsPlaying(false)}
                onError={(e) => {
                  console.error('Audio error:', e);
                  setError('Error playing audio: ' + e.target.error?.message || 'Unknown error');
                }}
              />
              <Button
                onClick={handlePlayPause}
                className="w-full bg-fuchsia-100 text-fuchsia-800 hover:bg-fuchsia-200"
              >
                {isPlaying ? (
                  <><Pause className="mr-2" size={18} /> Pause</>
                ) : (
                  <><Play className="mr-2" size={18} /> Play Translation</>
                )}
              </Button>
            </div>
          )}
        </CardContent>
        <CardFooter>
          <Button 
            onClick={processAudio} 
            disabled={!file || !targetLanguage || processing}
            className="w-full bg-gradient-to-r from-fuchsia-600 to-pink-600 hover:from-fuchsia-700 hover:to-pink-700 text-white"
          >
            {processing ? progressText : 'Translate Audio'}
          </Button>
        </CardFooter>
      </Card>
    </div>
  );
};

export default LinguaSyncApp;