import React from "react";
import { NavigationContainer } from "@react-navigation/native";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import MissionScreen from "./src/screens/MissionScreen";
import DrawingScreen from "./src/screens/DrawingScreen";

const Stack = createNativeStackNavigator();

export default function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator initialRouteName="Mission">
        <Stack.Screen
          name="Mission"
          component={MissionScreen}
          options={{ headerShown: false }}
        />
        <Stack.Screen
          name="Drawing"
          component={DrawingScreen}
          options={({ route }) => ({
            title: route.params?.displayText || "Draw",
            headerShown: false,
          })}
        />
      </Stack.Navigator>
    </NavigationContainer>
  );
}
