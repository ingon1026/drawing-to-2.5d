import React from "react";
import {
  View,
  Text,
  FlatList,
  TouchableOpacity,
  StyleSheet,
  SafeAreaView,
} from "react-native";
import missions from "../data/missions.json";

export default function MissionScreen({ navigation }) {
  const renderItem = ({ item }) => (
    <TouchableOpacity
      style={styles.card}
      activeOpacity={0.7}
      onPress={() =>
        navigation.navigate("Drawing", {
          category: item.id,
          displayText: item.displayText,
        })
      }
    >
      <Text style={styles.emoji}>{item.emoji}</Text>
      <Text style={styles.label}>{item.label}</Text>
    </TouchableOpacity>
  );

  return (
    <SafeAreaView style={styles.container}>
      <Text style={styles.title}>Sketch Guide</Text>
      <Text style={styles.subtitle}>Choose what to draw</Text>
      <FlatList
        data={missions}
        renderItem={renderItem}
        keyExtractor={(item) => item.id}
        numColumns={3}
        contentContainerStyle={styles.grid}
        columnWrapperStyle={styles.row}
      />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#ffffff",
  },
  title: {
    fontSize: 28,
    fontWeight: "bold",
    textAlign: "center",
    marginTop: 20,
    color: "#333",
  },
  subtitle: {
    fontSize: 16,
    textAlign: "center",
    marginTop: 4,
    marginBottom: 20,
    color: "#999",
  },
  grid: {
    paddingHorizontal: 16,
    paddingBottom: 20,
  },
  row: {
    justifyContent: "space-between",
    marginBottom: 12,
  },
  card: {
    flex: 1,
    maxWidth: "31%",
    aspectRatio: 1,
    backgroundColor: "#f8f9fa",
    borderRadius: 16,
    alignItems: "center",
    justifyContent: "center",
    marginHorizontal: 4,
    borderWidth: 1,
    borderColor: "#e8e8e8",
  },
  emoji: {
    fontSize: 36,
    marginBottom: 6,
  },
  label: {
    fontSize: 13,
    fontWeight: "600",
    color: "#555",
  },
});
