export default {
	async fetch(request, env) {
		if (request.method === 'OPTIONS') {
			return new Response(null, {
				headers: {
					'Access-Control-Allow-Origin': '*',
					'Access-Control-Allow-Methods': 'POST',
					'Access-Control-Allow-Headers': 'Content-Type',
				},
			});
		}

		if (request.method !== 'POST') {
			return new Response('Please send a POST request', { status: 405 });
		}

		try {
			const { image, style_strength } = await request.json();

			if (!image) {
				return new Response('Please provide a base64 image', { status: 400 });
			}

			// Convert base64 image to Uint8Array
			const binaryString = atob(image.split(',')[1]);
			const len = binaryString.length;
			const bytes = new Uint8Array(len);
			for (let i = 0; i < len; i++) {
				bytes[i] = binaryString.charCodeAt(i);
			}

			// Steampunk style prompt
			// const prompt = `Transform this image into a steampunk style while keeping the main subject clearly recognizable.`;

			// Steampunk style prompt
			const prompt = `Transform this image into a steampunk style while keeping the main subject clearly recognizable. 
Use a color palette of brass, copper, brown, and deep burgundy tones.`;

			// Call Stable Diffusion with img2img
			const aiResponse = await env.AI.run('@cf/runwayml/stable-diffusion-v1-5-img2img', {
				prompt: prompt,
				image: [...bytes],
				num_inference_steps: 50,
				strength: style_strength || 0.75, // Adjustable style strength
				guidance_scale: 7.5,
			});

			// Return the transformed image
			return new Response(aiResponse, {
				headers: {
					'Content-Type': 'image/png',
					'Access-Control-Allow-Origin': '*',
				},
			});
		} catch (error) {
			console.log({ error });
			return new Response(`Error: ${error.message}`, {
				status: 500,
				headers: {
					'Access-Control-Allow-Origin': '*',
				},
			});
		}
	},
};
