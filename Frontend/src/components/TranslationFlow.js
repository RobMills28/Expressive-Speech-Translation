import React, { useState, useRef, useCallback } from 'react';
import { Card, CardContent } from "./ui/card";
import { Button } from "./ui/button";
import { Progress } from "./ui/progress";
import { Upload, Globe, Play, Pause, CheckCircle, AlertCircle } from 'lucide-react';
import { useTranslation } from '../hooks/useTranslation';
import { Alert, AlertDescription, AlertTitle } from "./ui/alert";

const TranslationFlow = () => {
  // States from the translation hook
  const {
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
    audioRef,
    handleFileChange,
    handleLanguageSelect,
    handlePlayPause,
    processAudio,
    cleanup,
    setError,
    setAudioStatus,
    setAudioReady,
    setIsPlaying,
    setShowTranscript
  } = useTranslation();

  // Local states for the flow UI
  const [step, setStep] = useState(1);
  const [selectedLanguages, setSelectedLanguages] = useState([]);
  const [translations, setTranslations] = useState({});
  const [mediaType, setMediaType] = useState(null); // 'audio', 'video', or 'both'
  const fileInputRef = useRef(null);

  const languages = [
    { code: 'fra', name: 'French', flag: '🇫🇷' },
    { code: 'spa', name: 'Spanish', flag: '🇪🇸' },
    { code: 'deu', name: 'German', flag: '🇩🇪' },
    { code: 'ita', name: 'Italian', flag: '🇮🇹' },
    { code: 'por', name: 'Portuguese', flag: '🇵🇹' }
  ];

  const handleMediaTypeSelect = (type) => {
    setMediaType(type);
    setStep(2);
  };

  const handleFileUpload = useCallback((event) => {
    handleFileChange(event);  // Use the hook's file handler
    if (event.target.files[0]) {
      setStep(3);
    }
  }, [handleFileChange]);

  const handleLanguageToggle = useCallback((code) => {
    setSelectedLanguages(prev => {
      if (prev.includes(code)) {
        return prev.filter(c => c !== code);
      }
      return [...prev, code];
    });
    handleLanguageSelect(code);  // Update the hook's target language
  }, [handleLanguageSelect]);

  const handleContinue = useCallback(async () => {
    if (selectedLanguages.length > 0) {
      setStep(4);
      try {
        await processAudio();  // Use the hook's process function
        setTranslations(prev => ({
          ...prev,
          [targetLanguage]: {
            audioUrl: translatedAudioUrl,
            sourceText,
            targetText
          }
        }));
      } catch (e) {
        setError(e.message);
      }
    }
  }, [selectedLanguages, processAudio, targetLanguage, translatedAudioUrl, sourceText, targetText, setError]);

  const handleBackStep = () => {
    if (step > 1) {
      setStep(step - 1);
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto p-4">
      {/* Progress Steps - Perfectly Centered */}
      <div className="flex justify-center mb-12">
        <div className="flex items-center justify-between" style={{ width: '280px' }}>
          {[1, 2, 3].map((number) => (
            <div key={number} className="flex items-center">
              <div className={`
                w-10 h-10 rounded-full flex items-center justify-center
                ${step > number ? 'bg-green-500 text-white' : 
                  step === number ? 'bg-fuchsia-600 text-white' : 
                  'bg-gray-200 text-gray-600'}
              `}>
                {step > number ? <CheckCircle className="w-5 h-5" /> : number}
              </div>
              {number < 3 && (
                <div className={`w-16 h-1 mx-1 ${
                  step > number ? 'bg-green-500' : 'bg-gray-200'
                }`} />
              )}
            </div>
          ))}
        </div>
      </div>

      <Card className="bg-white shadow-lg">
        <CardContent className="p-6">
          {/* Error Display */}
          {error && (
            <Alert variant="destructive" className="mb-4">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>Error</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {/* Step 1: Choose Content Type */}
          {step === 1 && (
            <div className="text-center">
              <h2 className="text-xl font-semibold mb-6">Choose Content Type</h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div 
                  className="border rounded-lg p-6 cursor-pointer hover:border-fuchsia-500 hover:bg-fuchsia-50 transition-colors"
                  onClick={() => handleMediaTypeSelect('audio')}
                >
                  <div className="bg-fuchsia-100 w-12 h-12 rounded-full flex items-center justify-center mx-auto mb-4">
                    <Upload className="w-6 h-6 text-fuchsia-600" />
                  </div>
                  <h3 className="font-medium mb-2">Audio Translation</h3>
                  <p className="text-sm text-gray-500">Translate podcasts and audio content</p>
                </div>
                <div 
                  className="border rounded-lg p-6 cursor-pointer hover:border-fuchsia-500 hover:bg-fuchsia-50 transition-colors"
                  onClick={() => handleMediaTypeSelect('video')}
                >
                  <div className="bg-fuchsia-100 w-12 h-12 rounded-full flex items-center justify-center mx-auto mb-4">
                    <Upload className="w-6 h-6 text-fuchsia-600" />
                  </div>
                  <h3 className="font-medium mb-2">Video Translation</h3>
                  <p className="text-sm text-gray-500">Translate videos with synchronized subtitles</p>
                </div>
                <div 
                  className="border rounded-lg p-6 cursor-pointer hover:border-fuchsia-500 hover:bg-fuchsia-50 transition-colors"
                  onClick={() => handleMediaTypeSelect('both')}
                >
                  <div className="bg-fuchsia-100 w-12 h-12 rounded-full flex items-center justify-center mx-auto mb-4">
                    <Upload className="w-6 h-6 text-fuchsia-600" />
                  </div>
                  <h3 className="font-medium mb-2">Audio + Video</h3>
                  <p className="text-sm text-gray-500">Translate both audio and visual elements</p>
                </div>
              </div>
            </div>
          )}

          {/* Step 2: Upload */}
          {step === 2 && (
            <div className="text-center">
              <div 
                className="border-2 border-dashed border-gray-300 rounded-lg p-12 cursor-pointer hover:border-fuchsia-500 transition-colors"
                onClick={() => fileInputRef.current?.click()}
              >
                <Upload className="w-12 h-12 mx-auto mb-4 text-fuchsia-600" />
                <h2 className="text-xl font-semibold mb-2">Upload Your Content</h2>
                <p className="text-gray-600 mb-4">
                  {mediaType === 'audio' ? 'Drag and drop your audio file here' :
                   mediaType === 'video' ? 'Drag and drop your video file here' :
                   'Drag and drop your audio or video file here'}
                </p>
                <Button className="bg-fuchsia-600 hover:bg-fuchsia-700">
                  Select File
                </Button>
                <input
                  ref={fileInputRef}
                  type="file"
                  className="hidden"
                  onChange={handleFileUpload}
                  accept={mediaType === 'audio' ? 'audio/*' : 
                          mediaType === 'video' ? 'video/*' : 
                          'audio/*,video/*'}
                />
              </div>
              <div className="mt-4 flex justify-start">
                <Button 
                  variant="outline" 
                  onClick={handleBackStep}
                  className="text-fuchsia-600 border-fuchsia-200 hover:bg-fuchsia-50"
                >
                  Back
                </Button>
              </div>
            </div>
          )}

          {/* Step 3: Language Selection */}
          {step === 3 && (
            <div>
              <h2 className="text-xl font-semibold mb-4">Choose Languages</h2>
              <div className="space-y-3 mb-6">
                {languages.map((language) => (
                  <label
                    key={language.code}
                    className={`flex items-center p-3 border rounded-lg hover:bg-gray-50 cursor-pointer ${
                      selectedLanguages.includes(language.code) ? 'border-fuchsia-300 bg-fuchsia-50' : 'border-gray-200'
                    }`}
                  >
                    <input
                      type="checkbox"
                      className="w-5 h-5 text-fuchsia-600 border-fuchsia-300 focus:ring-fuchsia-500"
                      checked={selectedLanguages.includes(language.code)}
                      onChange={() => handleLanguageToggle(language.code)}
                    />
                    <Globe className={`w-5 h-5 ml-3 mr-2 ${
                      selectedLanguages.includes(language.code) ? 'text-fuchsia-500' : 'text-gray-400'
                    }`} />
                    <span>{language.flag} {language.name}</span>
                  </label>
                ))}
              </div>
              <div className="flex justify-between">
                <Button 
                  variant="outline" 
                  onClick={handleBackStep}
                  className="text-fuchsia-600 border-fuchsia-200 hover:bg-fuchsia-50"
                >
                  Back
                </Button>
                <Button
                  onClick={handleContinue}
                  disabled={selectedLanguages.length === 0 || processing}
                  className="bg-fuchsia-600 hover:bg-fuchsia-700 disabled:bg-gray-300"
                >
                  {processing ? 'Processing...' : 'Continue'}
                </Button>
              </div>
            </div>
          )}

          {/* Step 4: Preview Translations */}
          {step === 4 && (
            <div>
              <h2 className="text-xl font-semibold mb-4">Preview Translations</h2>
              
              {/* Hidden audio element */}
              <audio 
                ref={audioRef} 
                src={translatedAudioUrl}
                className="hidden"
                preload="auto"
                onLoadStart={() => {
                  setAudioReady(false);
                  setAudioStatus('loading');
                }}
                onCanPlay={() => {
                  setAudioStatus('ready');
                  setAudioReady(true);
                  setError('');
                }}
                onEnded={() => {
                  setIsPlaying(false);
                  setAudioStatus('ready');
                }}
                onError={(e) => {
                  console.error('Audio error:', e);
                  setError('Failed to load audio');
                  setAudioStatus('error');
                  setAudioReady(false);
                  setIsPlaying(false);
                }}
              />
              
              {/* Original Audio */}
              <div className="mb-6">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="font-medium">Original</h3>
                  {sourceText && (
                    <p className="text-sm text-gray-600">{sourceText}</p>
                  )}
                </div>
                <div className="bg-gray-100 h-12 rounded-lg flex items-center justify-center">
                  {file && <p className="text-sm text-gray-600">{file.name}</p>}
                </div>
              </div>

              {/* Translated Audio */}
              {selectedLanguages.map(code => {
                const language = languages.find(l => l.code === code);
                const translation = translations[code];
                
                return (
                  <div key={code} className="mb-6">
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="font-medium">{language?.name}</h3>
                      <Button 
                        size="sm" 
                        variant="outline"
                        className="border-fuchsia-200 text-fuchsia-700 hover:bg-fuchsia-50"
                        onClick={handlePlayPause}
                        disabled={!audioReady || audioStatus === 'error'}
                      >
                        {isPlaying ? 
                          <Pause className="w-4 h-4" /> : 
                          <Play className="w-4 h-4" />
                        }
                      </Button>
                    </div>
                    <div className="bg-gray-100 h-12 rounded-lg flex items-center justify-center">
                      {translation?.targetText && (
                        <p className="text-sm text-gray-600">{translation.targetText}</p>
                      )}
                    </div>
                  </div>
                );
              })}

              <div className="flex justify-between">
                <Button 
                  variant="outline"
                  className="text-fuchsia-600 border-fuchsia-200 hover:bg-fuchsia-50"
                  onClick={handleBackStep}
                >
                  Back
                </Button>
                <Button 
                  className="bg-fuchsia-600 hover:bg-fuchsia-700"
                  onClick={() => {
                    cleanup();
                    setStep(1);
                  }}
                >
                  Start New Translation
                </Button>
              </div>
            </div>
          )}

          {/* Processing Indicator */}
          {processing && (
            <div className="mt-4">
              <Progress value={progress} className="mb-2 bg-gray-200 [&>div]:bg-fuchsia-600" />
              <p className="text-sm text-gray-600 text-center">
                {progressText || 'Processing translations...'}
              </p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default TranslationFlow;