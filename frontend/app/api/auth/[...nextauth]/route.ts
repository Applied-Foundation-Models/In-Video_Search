import NextAuth from 'next-auth/next';
import GoogleProvider from "next-auth/providers/google";

// Ensure the environment variables are defined
const googleClientId = process.env.GOOGLE_CLIENT_ID;
const googleClientSecret = process.env.GOOGLE_CLIENT_SECRET;

if (!googleClientId || !googleClientSecret) {
    throw new Error("Google Client ID and Secret must be set in environment variables");
}

const handler = NextAuth({
    providers: [
        GoogleProvider({
            clientId: googleClientId,
            clientSecret: googleClientSecret,
        }),
    ],
});

export { handler as GET, handler as POST };
