// App.js
import React, { useState, useRef, useEffect } from 'react';
import { StyleSheet, Text, View, Button, ActivityIndicator, ScrollView, Dimensions, Platform, TouchableOpacity } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { Video } from 'expo-av';
import { Picker } from '@react-native-picker/picker';

// --- IMPORTANT: REPLACE WITH YOUR COMPUTER'S LOCAL IP ADDRESS ---
// Your backend must be running and accessible on your local network at this IP and port
const BACKEND_URL = 'http://10.130.222.2:5001';
const APP_TITLE = "Content Converter AI";

const LANGUAGES = {
  'eng': { name: 'English', flag: '🇺🇸' },
  'fra': { name: 'French', flag: '🇫🇷' },
  'spa': { name: 'Spanish', flag: '🇪🇸' },
  'deu': { name: 'German', flag: '🇩🇪' },
  // Add other languages your backend supports
};

const { width } = Dimensions.get('window');
const videoWidth = width * 0.9;

export default function App() {
  const [video, setVideo] = useState(null);
  const [targetLanguage, setTargetLanguage] = useState('fra');
  const [isProcessing, setIsProcessing] = useState(false);
  const [progressMessage, setProgressMessage] = useState('');
  const [error, setError] = useState('');
  const [translatedVideoUri, setTranslatedVideoUri] = useState(null);

  const originalVideoPlayer = useRef(null);
  const translatedVideoPlayer = useRef(null);

  useEffect(() => {
    (async () => {
      if (Platform.OS !== 'web') {
        const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
        if (status !== 'granted') {
          alert('Sorry, we need camera roll permissions to make this work!');
        }
      }
    })();
  }, []);

  const pickVideo = async () => {
    setError('');
    setTranslatedVideoUri(null);
    setProgressMessage('');
    setVideo(null); // Reset previous video
    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Videos,
      allowsEditing: false,
      quality: 1,
    });

    if (!result.canceled && result.assets && result.assets.length > 0) {
      setVideo(result.assets[0]);
    } else {
      setVideo(null);
    }
  };

  const handleTranslate = async () => {
    if (!video) {
      setError('Please select a video first.');
      return;
    }
    if (!targetLanguage) {
      setError('Please select a target language.');
      return;
    }

    setIsProcessing(true);
    setError('');
    setTranslatedVideoUri(null);
    setProgressMessage('Starting upload...');

    const formData = new FormData();
    formData.append('video', {
      uri: video.uri,
      name: video.fileName || `video-${Date.now()}.${video.uri.split('.').pop()}`,
      type: video.mimeType || `video/${video.uri.split('.').pop()}`,
    });
    formData.append('target_language', targetLanguage);
    formData.append('apply_lip_sync', 'false'); 
    formData.append('use_voice_cloning', 'true');

    try {
      const response = await fetch(`${BACKEND_URL}/process-video`, {
        method: 'POST',
        body: formData,
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({ error: `Server error: ${response.status} ${response.statusText}` }));
        throw new Error(errData.error || `Request failed: ${response.statusText}`);
      }
      
      // Simplified result handling (assumes final response is the video or JSON with base64)
      // YOU WILL LIKELY NEED TO ADAPT THIS TO YOUR BACKEND'S SSE IMPLEMENTATION
      const contentType = response.headers.get('content-type');
      if (contentType && contentType.includes('video/mp4')) {
        const blob = await response.blob();
        const reader = new FileReader();
        reader.readAsDataURL(blob);
        reader.onloadend = () => {
            setTranslatedVideoUri(reader.result);
            setProgressMessage('Translation complete!');
        };
        reader.onerror = () => { setError('Failed to read translated video.'); setProgressMessage(''); };
      } else {
        // Attempt to parse as JSON if not a direct video stream
        const data = await response.json();
        if (data.error) {
            throw new Error(data.error);
        }
        // This assumes your backend's SSE, IF USED for the final result, would put it in a 'result' field
        // and that field would contain base64 video data. This part is very speculative without knowing
        // your exact SSE structure for the final video payload.
        if (data.result) { // Check if data.result (base64 video) exists
            const videoDataUri = `data:video/mp4;base64,${data.result}`;
            setTranslatedVideoUri(videoDataUri);
            setProgressMessage('Translation complete!');
        } else if (data.phase && data.phase.includes("Video ready!")) { // If SSE gave a phase indicating completion
             // This assumes the video URI might have been set by an earlier SSE message
             // Or you need another mechanism to get the final video URI/data
             setProgressMessage('Video processed, awaiting final data...');
             // You might need to make another call or have the backend push the final URL/data
        } else {
            // If the response is JSON but doesn't contain an error or the expected video data
            console.log("Received JSON response without 'result' or 'error':", data);
            setProgressMessage('Processing complete, but video data not found in response.');
            // setError('Video processed, but could not display. Check backend response format.');
        }
      }
    } catch (err) {
      console.error("Translation error:", err);
      setError(err.message || 'An unknown error occurred.');
      setProgressMessage('Failed.');
    } finally {
      setIsProcessing(false);
    }
  };

  const renderVideoPlayer = (uri, playerRef, placeholderText) => {
    if (uri) {
      return (
        <Video
          ref={playerRef}
          style={styles.videoPlayer}
          source={{ uri: uri }}
          useNativeControls
          resizeMode="contain"
          onError={(e) => { console.error("Video Player Error:", e); setError('Error playing video: ' + e.error); }}
        />
      );
    }
    return <Text style={styles.placeholderText}>{placeholderText}</Text>;
  };

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>{APP_TITLE}</Text>

      <TouchableOpacity style={styles.button} onPress={pickVideo} disabled={isProcessing}>
        <Text style={styles.buttonText}>Pick a Video from Library</Text>
      </TouchableOpacity>

      {video && (
        <View style={styles.videoSection}>
          <Text style={styles.videoLabel}>Original Video:</Text>
          {renderVideoPlayer(video.uri, originalVideoPlayer, "No video selected")}
          <Text style={styles.filePathText}>Selected: {video.fileName || video.uri.split('/').pop()}</Text>
        </View>
      )}

      <View style={styles.pickerContainer}>
        <Text style={styles.label}>Target Language:</Text>
        <View style={styles.pickerWrapper}>
            <Picker
              selectedValue={targetLanguage}
              style={styles.picker}
              onValueChange={(itemValue) => setTargetLanguage(itemValue)}
              enabled={!isProcessing}
              itemStyle={styles.pickerItem} // For iOS text color
            >
              {Object.entries(LANGUAGES).map(([code, { name, flag }]) => (
                <Picker.Item key={code} label={`${flag} ${name}`} value={code} />
              ))}
            </Picker>
        </View>
      </View>

      <TouchableOpacity 
        style={[styles.button, styles.translateButton, (!video || isProcessing || !targetLanguage) && styles.buttonDisabled]} 
        onPress={handleTranslate} 
        disabled={!video || isProcessing || !targetLanguage}
      >
        <Text style={styles.buttonText}>
            {isProcessing ? 'Processing...' : 'Translate Video'}
        </Text>
      </TouchableOpacity>


      {isProcessing && (
        <View style={styles.processingContainer}>
          <ActivityIndicator size="large" color="#4A00E0" />
          <Text style={styles.progressText}>{progressMessage || 'Processing...'}</Text>
        </View>
      )}

      {error && (
        <Text style={styles.errorText}>Error: {error}</Text>
      )}

      {translatedVideoUri && (
        <View style={styles.videoSection}>
          <Text style={styles.videoLabel}>Translated Video ({LANGUAGES[targetLanguage]?.name}):</Text>
          {renderVideoPlayer(translatedVideoUri, translatedVideoPlayer, "Translation will appear here")}
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flexGrow: 1,
    backgroundColor: '#F3F4F6', // Light gray background
    alignItems: 'center',
    paddingVertical: 30,
    paddingHorizontal: 15,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    marginBottom: 25,
    color: '#374151', // Darker gray
    textAlign: 'center',
  },
  button: {
    backgroundColor: '#8E2DE2', // Purple gradient start
    paddingVertical: 12,
    paddingHorizontal: 30,
    borderRadius: 25,
    marginVertical: 10,
    width: '90%',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 4,
    elevation: 3,
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  translateButton: {
    backgroundColor: '#4A00E0', // Darker purple for translate
  },
  buttonDisabled: {
    backgroundColor: '#D1D5DB', // Gray when disabled
  },
  videoSection: {
    marginVertical: 20,
    width: videoWidth,
    alignItems: 'center',
    backgroundColor: '#FFFFFF',
    padding: 15,
    borderRadius: 12,
    borderColor: '#E5E7EB',
    borderWidth: 1,
  },
  videoLabel: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 12,
    color: '#1F2937', // Even darker gray
  },
  videoPlayer: {
    width: videoWidth - 30, 
    height: (videoWidth - 30) * (9 / 16), 
    backgroundColor: '#000000',
    borderRadius: 8,
  },
  filePathText: {
    fontSize: 12,
    color: '#6B7280', // Medium gray
    marginTop: 8,
    textAlign: 'center',
  },
  pickerContainer: {
    marginVertical: 20,
    width: '90%',
  },
  label: {
    fontSize: 16,
    marginBottom: 8,
    color: '#374151',
    alignSelf: 'flex-start',
  },
  pickerWrapper: { // Wrapper for Android picker styling
    backgroundColor: '#FFFFFF',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#D1D5DB',
    overflow: 'hidden', // Important for border radius on Android
  },
  picker: {
    height: Platform.OS === 'ios' ? 120 : 50, // iOS needs more height for wheel
    width: '100%',
    color: '#1F2937', // For Android text color
  },
  pickerItem: { // For iOS picker item text color
    color: '#1F2937',
  },
  processingContainer: {
    marginTop: 25,
    alignItems: 'center',
  },
  progressText: {
    marginTop: 12,
    fontSize: 16,
    color: '#4B5563',
  },
  errorText: {
    marginTop: 20,
    color: '#EF4444', // Red for errors
    fontSize: 16,
    textAlign: 'center',
    paddingHorizontal: 10,
  },
  placeholderText: {
    textAlign: 'center',
    color: '#9CA3AF', // Lighter gray for placeholders
    fontStyle: 'italic',
    paddingVertical: 60,
    fontSize: 16,
  }
});