"use client";
import React, { useEffect, useState } from "react";
import { initializeApp } from "firebase/app";
import { getAuth, GoogleAuthProvider, signInWithPopup, signOut, User } from "firebase/auth";

const firebaseConfig = {
  apiKey: "AIzaSyAWG3QTgAZD0auuYSiosFnqc_ZPH1qPUDQ",
  authDomain: "login-f97c0.firebaseapp.com",
  projectId: "login-f97c0",
  storageBucket: "login-f97c0.appspot.com",
  messagingSenderId: "307880516676",
  appId: "1:307880516676:web:ef67dff7f21463c7b3f011",
  measurementId: "G-GT63FRXP8C"
};

const LoginPage = () => {
  const [auth, setAuth] = useState<ReturnType<typeof getAuth> | null>(null);
  const [provider, setProvider] = useState<GoogleAuthProvider | null>(null);
  const [user, setUser] = useState<User | null>(null);

  useEffect(() => {
    // Only run on client-side
    if (typeof window !== 'undefined') {
      const app = initializeApp(firebaseConfig);
      const authInstance = getAuth(app);
      const providerInstance = new GoogleAuthProvider();

      setAuth(authInstance);
      setProvider(providerInstance);

      // Optional: Set up an auth state listener
      const unsubscribe = authInstance.onAuthStateChanged((currentUser) => {
        setUser(currentUser);
      });

      // Cleanup subscription on unmount
      return () => unsubscribe();
    }
  }, []);

  const handleSignIn = async () => {
    if (!auth || !provider) {
      console.error('Firebase not initialized');
      return;
    }

    try {
      const result = await signInWithPopup(auth, provider);
      const credential = GoogleAuthProvider.credentialFromResult(result);

      if (credential) {
        const token = credential.accessToken;
        const user = result.user;
        console.log('User:', user);
        console.log('Token:', token);
      } else {
        console.error('No credentials received.');
      }
    } catch (error) {
      console.error('Error during sign-in:', error);
    }
  };

  const handleSignOut = async () => {
    if (!auth) {
      console.error('Firebase not initialized');
      return;
    }

    try {
      await signOut(auth);
      console.log("User signed out successfully.");
    } catch (error) {
      console.error("Error during sign-out:", error);
    }
  };

  return (
    <main className="flex flex-col items-center space-y-4 mt-4">
      {!user ? (
        <button 
          onClick={handleSignIn} 
          disabled={!auth}
          className="bg-gray-100 p-2 rounded-md shadow disabled:opacity-50"
        >
          Sign In with Google
        </button>
      ) : (
        <>
          <div className="text-xl">Welcome, {user.displayName}</div>
          <button 
            onClick={handleSignOut} 
            className="bg-gray-100 p-2 rounded-md shadow"
          >
            Sign Out
          </button>
        </>
      )}
    </main>
  );
};

export default LoginPage;