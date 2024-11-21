import { initializeApp } from "firebase/app";
import { getAuth, GoogleAuthProvider } from 'firebase/auth';

const firebaseConfig = {
  apiKey: "AIzaSyAWG3QTgAZD0auuYSiosFnqc_ZPH1qPUDQ",
  authDomain: "login-f97c0.firebaseapp.com",
  projectId: "login-f97c0",
  storageBucket: "login-f97c0.appspot.com",
  messagingSenderId: "307880516676",
  appId: "1:307880516676:web:ef67dff7f21463c7b3f011",
  measurementId: "G-GT63FRXP8C"
};

let app;
if (typeof window !== 'undefined') {
  app = initializeApp(firebaseConfig);
}

const auth = getAuth(app);
const provider = new GoogleAuthProvider();

export { auth, provider  };