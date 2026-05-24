import asyncio
import httpx
import time

API_URL = "https://web-production-92e5f.up.railway.app"
API_KEY = "farmerxential-secret-key-2024"
HEADERS = {"X-API-Key": API_KEY}

async def call_stats(client, call_number):
    try:
        start = time.time()
        response = await client.get(f"{API_URL}/stats", headers=HEADERS)
        duration = round((time.time() - start) * 1000, 1)
        return {
            "call": call_number,
            "status": response.status_code,
            "duration_ms": duration,
            "success": response.status_code == 200
        }
    except Exception as e:
        return {
            "call": call_number,
            "status": "ERROR",
            "duration_ms": 0,
            "success": False,
            "error": str(e)
        }

async def run_load_test():
    print("Starting FarmerXential load test — 100 concurrent calls...")
    print("=" * 50)

    async with httpx.AsyncClient(timeout=30) as client:
        start_total = time.time()

        tasks = [call_stats(client, i+1) for i in range(100)]
        results = await asyncio.gather(*tasks)

        total_time = round((time.time() - start_total) * 1000, 1)

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    durations = [r["duration_ms"] for r in successful]

    print(f"Total calls: 100")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Success rate: {len(successful)}%")
    print(f"")
    print(f"Response times:")
    print(f"  Fastest: {min(durations):.1f}ms")
    print(f"  Slowest: {max(durations):.1f}ms")
    print(f"  Average: {sum(durations)/len(durations):.1f}ms")
    print(f"")
    print(f"Total test duration: {total_time}ms")

    if failed:
        print(f"\nFailed calls:")
        for f in failed[:5]:
            print(f"  Call {f['call']}: {f.get('error', f['status'])}")

    if len(successful) == 100:
        print("\n✅ PASSED — FarmerXential API handles 100 concurrent calls!")
    elif len(successful) >= 90:
        print("\n⚠️ PARTIAL — API handles most calls but some failed")
    else:
        print("\n❌ FAILED — API struggles under load")

asyncio.run(run_load_test())