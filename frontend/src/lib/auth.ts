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

    server: 'sandbox'
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
                            productId: "afdfc695-519e-47a1-a469-918d28e6c4c8", 
                            slug: "pro" 
                        },
                        { 
                            productId: "13a83be1-0727-4f28-aec4-bd5d935d8ed0", 
                            slug: "premium" 
                        },
                        { 
                            productId: "cec41d08-7a3b-41df-bc2d-f756cc7eacc3", 
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
                    case "afdfc695-519e-47a1-a469-918d28e6c4c8":
                        creditsToAdd = 20;
                        break;
                    case "13a83be1-0727-4f28-aec4-bd5d935d8ed0":
                        creditsToAdd = 60;
                        break;
                    case "cec41d08-7a3b-41df-bc2d-f756cc7eacc3":
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