import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "./components/ui/card"
import { Button } from "./components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./components/ui/select"
import { Label } from "./components/ui/label"
import { Input } from "./components/ui/input"
import { AlertCircle, Globe, Mic, Download } from 'lucide-react'
import { Alert, AlertDescription, AlertTitle } from "./components/ui/alert"
import { Progress } from "./components/ui/progress"

const LinguaSyncApp = () => {
  const [file, setFile] = useState(null);
  const [targetLanguage, setTargetLanguage] = useState('');
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState('');
  const [translatedAudioUrl, setTranslatedAudioUrl] = useState('');

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
    setError('');
    setTranslatedAudioUrl('');
  };

  const handleLanguageChange = (value) => {
    setTargetLanguage(value);
    setError('');
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
      const response = await fetch('http://localhost:5000/translate', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }

      const blob = await response.blob();
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
            <Input id="audio" type="file" accept="audio/*" onChange={handleFileChange} className="mt-1 bg-white border-fuchsia-300" />
          </div>
          <div>
            <Label htmlFor="language" className="text-sm font-medium text-fuchsia-800 flex items-center">
              <Globe className="mr-2" size={18} />
              Target Language
            </Label>
            <Select value={targetLanguage} onValueChange={handleLanguageChange}>
              <SelectTrigger className="w-full mt-1 border-fuchsia-300">
                <SelectValue placeholder="Select language" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="fr">🇫🇷 French</SelectItem>
                <SelectItem value="es">🇪🇸 Spanish</SelectItem>
                <SelectItem value="de">🇩🇪 German</SelectItem>
                <SelectItem value="zh">🇨🇳 Chinese</SelectItem>
                <SelectItem value="ja">🇯🇵 Japanese</SelectItem>
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
            <div className="space-y-2">
              <audio controls src={translatedAudioUrl} className="w-full" />
              <Button
                onClick={() => {
                  const a = document.createElement('a');
                  a.href = translatedAudioUrl;
                  a.download = 'translated_audio.mp3';
                  document.body.appendChild(a);
                  a.click();
                  document.body.removeChild(a);
                }}
                className="w-full flex items-center justify-center"
              >
                <Download className="mr-2" size={18} />
                Download Translated Audio
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
            {processing ? 'Processing...' : 'Translate Audio'}
          </Button>
        </CardFooter>
      </Card>
    </div>
  );
};

export default LinguaSyncApp;