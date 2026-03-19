// ── PASTE YOUR FIREBASE CONFIG HERE ──────────────────────────────────────────
// Get this from: Firebase Console → Project Settings → Your apps → Web app
const firebaseConfig = {
  apiKey: "AIzaSyCpPjaaJzINkjoWiQ_cAR8O1_o5b7WeAms",
  authDomain: "risklens-2ebaa.firebaseapp.com",
  projectId: "risklens-2ebaa",
  storageBucket: "risklens-2ebaa.firebasestorage.app",
  messagingSenderId: "298187858580",
  appId: "1:298187858580:web:a9f2fe61db88b98c2e0a81",
};

// ── INIT ──────────────────────────────────────────────────────────────────────
import { initializeApp }                              from "https://www.gstatic.com/firebasejs/10.12.0/firebase-app.js";
import { getAuth, GoogleAuthProvider, signInWithPopup, signOut, onAuthStateChanged }
  from "https://www.gstatic.com/firebasejs/10.12.0/firebase-auth.js";

const app      = initializeApp(firebaseConfig);
const auth     = getAuth(app);
const provider = new GoogleAuthProvider();

// ── EXPORTS ───────────────────────────────────────────────────────────────────
export { auth, provider };

export const signInWithGoogle = () => signInWithPopup(auth, provider);

export const logOut = () => signOut(auth);

export const onAuth = (cb) => onAuthStateChanged(auth, cb);

export const getUser = () => auth.currentUser;