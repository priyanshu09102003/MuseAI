import { betterAuth} from "better-auth";
import { prismaAdapter } from "better-auth/adapters/prisma";
import { prisma } from "./prisma";
import { Polar } from "@polar-sh/sdk";
import {
  polar,
  checkout,
  portal,
  usage,
  webhooks,
} from "@polar-sh/better-auth";

const polarClient = new Polar({ 
    accessToken: process.env.POLAR_ACCESS_TOKEN, 

    server: 'production'
}); 


export const auth = betterAuth({
    database: prismaAdapter(prisma, {
        provider: "postgresql",
    }),
    emailAndPassword: {
        enabled: true
    },

    plugins: [
        polar({ 
            client: polarClient, 
            createCustomerOnSignUp: true, 
            use: [ 
                checkout({ 
                    products: [ 
                        { 
                            productId: "1d9c9bdd-f713-4b35-b712-92d744191b99", 
                            slug: "pro" 
                        },
                        { 
                            productId: "9ae76012-fdcc-4362-981b-69368912054f", 
                            slug: "premium" 
                        },
                        { 
                            productId: "1e617678-f389-499b-ae3c-4d94e5a03992", 
                            slug: "enterprise" 
                        },
                    ], 
                    successUrl: "/", 
                    authenticatedUsersOnly: true
                }), 
                portal(), 
                webhooks({ 
                    secret: process.env.POLAR_WEBHOOK_SECRET!,
                    onOrderPaid: async (order) => {
                    const externalCustomerId = order.data.customer.externalId;

                    if (!externalCustomerId) {
                    console.error("No external customer ID found.");
                    throw new Error("No external customer id found.");
                    }

                    const productId = order.data.productId;

                    let creditsToAdd = 0;

                    switch (productId) {
                    case "1d9c9bdd-f713-4b35-b712-92d744191b99":
                        creditsToAdd = 20;
                        break;
                    case "9ae76012-fdcc-4362-981b-69368912054f":
                        creditsToAdd = 60;
                        break;
                    case "1e617678-f389-499b-ae3c-4d94e5a03992":
                        creditsToAdd = 150;
                        break;
                    }

                    await prisma.user.update({
                        where: { id: externalCustomerId },
                        data: {
                            credits: {
                                increment: creditsToAdd,
                            },
                        },
                    });
                    }  
                })     
            ],         
        })             
    ]                  
});                    