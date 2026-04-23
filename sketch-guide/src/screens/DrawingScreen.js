import React, { useRef, useCallback, useEffect } from "react";
import { View, Text, StyleSheet, SafeAreaView } from "react-native";
import { WebView } from "react-native-webview";
import { getDrawingHTML } from "../utils/webContent";
import { getModelData } from "../utils/modelLoader";

export default function DrawingScreen({ route }) {
  const { category, displayText } = route.params;
  const webViewRef = useRef(null);
  const webViewReady = useRef(false);

  const sendModelToWebView = useCallback(async () => {
    if (!webViewRef.current) return;
    try {
      const modelData = getModelData(category);
      if (modelData) {
        const msg = JSON.stringify({
          action: "loadModel",
          category: category,
          modelData: modelData,
        });
        webViewRef.current.postMessage(msg);
      }
    } catch (e) {
      console.error("Failed to load model:", e);
    }
  }, [category]);

  const onWebViewMessage = useCallback(
    (event) => {
      try {
        const data = JSON.parse(event.nativeEvent.data);
        if (data.event === "ready") {
          webViewReady.current = true;
          sendModelToWebView();
        } else if (data.event === "modelLoaded") {
          console.log("Model loaded:", data.category);
        }
      } catch (e) {
        console.error("WebView message error:", e);
      }
    },
    [sendModelToWebView]
  );

  const htmlContent = getDrawingHTML();

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.missionText}>{displayText}</Text>
      </View>
      <View style={styles.webviewContainer}>
        <WebView
          ref={webViewRef}
          source={{ html: htmlContent }}
          style={styles.webview}
          originWhitelist={["*"]}
          javaScriptEnabled={true}
          domStorageEnabled={true}
          onMessage={onWebViewMessage}
          scrollEnabled={false}
          bounces={false}
          overScrollMode="never"
          showsHorizontalScrollIndicator={false}
          showsVerticalScrollIndicator={false}
        />
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#ffffff",
  },
  header: {
    paddingVertical: 12,
    paddingHorizontal: 16,
    backgroundColor: "#f8f9fa",
    borderBottomWidth: 1,
    borderBottomColor: "#e0e0e0",
    alignItems: "center",
  },
  missionText: {
    fontSize: 18,
    fontWeight: "600",
    color: "#4A90D9",
  },
  webviewContainer: {
    flex: 1,
  },
  webview: {
    flex: 1,
    backgroundColor: "#ffffff",
  },
});
