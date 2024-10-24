import React, { useState, useRef } from 'react';
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
  const audioRef = useRef(null);

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile) {
      console.log('Selected file:', selectedFile.name, 'Type:', selectedFile.type);
      
      // Check file extension
      const validExtensions = ['.mp3', '.wav', '.ogg', '.m4a'];
      const fileExtension = selectedFile.name.toLowerCase().slice(selectedFile.name.lastIndexOf('.'));
      
      // Check MIME type
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

      setFile(selectedFile);
      setError('');
      setTranslatedAudioUrl('');
    }
  };

  const handleLanguageChange = (value) => {
    console.log('Language selected:', value); // Debug log
    setTargetLanguage(value);
    setError('');
    setTranslatedAudioUrl('');
  };

  const handlePlayPause = () => {
    if (audioRef.current) {
      if (audioRef.current.paused) {
        audioRef.current.play();
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

    setProcessing(true);
    setProgress(0);
    setError('');
    setTranslatedAudioUrl('');

    const formData = new FormData();
    formData.append('file', file);
    formData.append('target_language', targetLanguage);

    try {
      console.log('Sending request with language:', targetLanguage); // Debug log

      const response = await fetch('http://localhost:5001/translate', {
        method: 'POST',
        body: formData,
      });

      // Log the raw response for debugging
      console.log('Response status:', response.status);
      const responseText = await response.text();
      console.log('Response text:', responseText);

      if (!response.ok) {
        let errorMessage;
        try {
          const errorData = JSON.parse(responseText);
          errorMessage = errorData.error;
        } catch (e) {
          errorMessage = `HTTP error! status: ${response.status}`;
        }
        throw new Error(errorMessage);
      }

      // Convert the response back to a blob
      const blob = new Blob([responseText], { type: 'audio/wav' });
      const url = URL.createObjectURL(blob);
      setTranslatedAudioUrl(url);
      setProgress(100);
    } catch (e) {
      console.error('Error details:', e);
      setError(`An error occurred: ${e.message}`);
    } finally {
      setProcessing(false);
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
            <Label htmlFor="audio" className="text-sm font-medium text-fuchsia-800 flex items-center">
              <Mic className="mr-2" size={18} />
              Upload Audio
            </Label>
            <div className="relative mt-1">
              <Input 
                id="audio" 
                type="file" 
                accept=".mp3,.wav,.ogg,.m4a,audio/*" 
                onChange={handleFileChange} 
                className="hidden"
              />
              <label 
                htmlFor="audio" 
                className="cursor-pointer inline-flex items-center px-4 py-2 bg-white border border-fuchsia-300 rounded-md font-semibold text-xs text-fuchsia-700 uppercase tracking-widest shadow-sm hover:bg-fuchsia-50 focus:outline-none focus:border-fuchsia-300 focus:ring focus:ring-fuchsia-200 active:bg-fuchsia-100 disabled:opacity-25 transition"
              >
                Choose file
              </label>
              <span className="ml-3 text-sm text-fuchsia-600 hover:text-fuchsia-800 cursor-default">
                {file ? file.name : "No file chosen"}
              </span>
            </div>
          </div>
          <div>
            <Label htmlFor="language" className="text-sm font-medium text-fuchsia-800 flex items-center">
              <Globe className="mr-2" size={18} />
              Target Language
            </Label>
            <Select 
              value={targetLanguage} 
              onValueChange={handleLanguageChange}
            >
              <SelectTrigger className="w-full mt-1 border-fuchsia-300 cursor-pointer hover:bg-fuchsia-50">
                <SelectValue placeholder="Select language" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="fr">🇫🇷 French</SelectItem>
                <SelectItem value="es">🇪🇸 Spanish</SelectItem>
                <SelectItem value="de">🇩🇪 German</SelectItem>
                <SelectItem value="it">🇮🇹 Italian</SelectItem>
                <SelectItem value="pt">🇵🇹 Portuguese</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {processing && (
            <div className="space-y-2">
              <Progress value={progress} className="w-full" />
              <p className="text-center text-sm text-fuchsia-800">Processing: {progress}%</p>
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
              <audio ref={audioRef} src={translatedAudioUrl} onEnded={() => setIsPlaying(false)} />
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
            className="w-full bg-gradient-to-r from-fuchsia-600 to-pink-600 hover:from-fuchsia-700 hover:to-pink-700 text-white cursor-default hover:cursor-pointer disabled:cursor-not-allowed"
          >
            {processing ? 'Processing...' : 'Translate Audio'}
          </Button>
        </CardFooter>
      </Card>
    </div>
  );
};

export default LinguaSyncApp;