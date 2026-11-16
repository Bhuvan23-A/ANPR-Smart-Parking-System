"""
Smart Parking Management and Automated Payment System
Integrates ANPR pipeline with:
1. SQLite Database for persistent parking logs, slots, and vehicle registrations
2. Real-time Slot Allocation Engine
3. Automated Entry Gate (Camera Trigger -> ANPR -> Slot Assignment -> Barrier OPEN)
4. Automated Exit Gate & Billing Engine (Duration calculation -> Tiered pricing -> Mall discount -> Payment)
5. FASTag / Digital Wallet / UPI QR receipt generation
"""

import os
import sqlite3
from datetime import datetime, timedelta

class SmartParkingSystem:
    def __init__(self, db_path="data/parking.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_db()

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def init_db(self):
        """Create database tables and populate seed parking slots & VIP whitelist."""
        conn = self.get_connection()
        cursor = conn.cursor()

        # 1. Registered Vehicles Whitelist (VIPs, Monthly Subscribers, Staff)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS registered_vehicles (
                plate_number TEXT PRIMARY KEY,
                owner_name TEXT NOT NULL,
                category TEXT NOT NULL,  -- 'VIP', 'MONTHLY_PASS', 'EMPLOYEE', 'REGULAR'
                registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                discount_percent REAL DEFAULT 0.0
            )
        """)

        # 2. Parking Slots
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS parking_slots (
                slot_id TEXT PRIMARY KEY,
                floor INTEGER NOT NULL,
                slot_type TEXT NOT NULL, -- 'GENERAL', 'VIP', 'EV_CHARGING'
                status TEXT DEFAULT 'AVAILABLE', -- 'AVAILABLE', 'OCCUPIED'
                current_plate TEXT
            )
        """)

        # 3. Parking Transactions / Logs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS parking_logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_number TEXT NOT NULL,
                entry_time TIMESTAMP NOT NULL,
                exit_time TIMESTAMP,
                slot_id TEXT NOT NULL,
                status TEXT DEFAULT 'ACTIVE', -- 'ACTIVE', 'COMPLETED'
                duration_minutes REAL DEFAULT 0,
                base_fee REAL DEFAULT 0,
                discount_amount REAL DEFAULT 0,
                final_amount REAL DEFAULT 0,
                payment_method TEXT,
                payment_status TEXT DEFAULT 'PENDING'
            )
        """)

        conn.commit()

        # Seed Parking Slots if empty
        cursor.execute("SELECT COUNT(*) FROM parking_slots")
        if cursor.fetchone()[0] == 0:
            slots = []
            # Floor 1: VIP (A-01 to A-05), EV (A-06 to A-10), General (A-11 to A-25)
            for i in range(1, 6):
                slots.append((f"P1-VIP-{i:02d}", 1, "VIP", "AVAILABLE", None))
            for i in range(6, 11):
                slots.append((f"P1-EV-{i:02d}", 1, "EV_CHARGING", "AVAILABLE", None))
            for i in range(11, 26):
                slots.append((f"P1-GEN-{i:02d}", 1, "GENERAL", "AVAILABLE", None))
            # Floor 2: General (P2-GEN-01 to P2-GEN-30)
            for i in range(1, 31):
                slots.append((f"P2-GEN-{i:02d}", 2, "GENERAL", "AVAILABLE", None))

            cursor.executemany("""
                INSERT INTO parking_slots (slot_id, floor, slot_type, status, current_plate)
                VALUES (?, ?, ?, ?, ?)
            """, slots)

        # Seed Registered Vehicles if empty
        cursor.execute("SELECT COUNT(*) FROM registered_vehicles")
        if cursor.fetchone()[0] == 0:
            registered = [
                ("DL01AB1234", "Dr. Rajesh Sharma", "VIP", 100.0),
                ("MH12DE4567", "Priya Verma", "MONTHLY_PASS", 100.0),
                ("KA05MJ9876", "Anand K.", "EMPLOYEE", 100.0),
                ("TN07BK3321", "Vikram Malhotra", "VIP", 100.0),
                ("TS09FA2024", "Sneha Rao", "MONTHLY_PASS", 50.0),
                ("DL08CZ5544", "Mall Operations Staff", "EMPLOYEE", 100.0)
            ]
            cursor.executemany("""
                INSERT INTO registered_vehicles (plate_number, owner_name, category, discount_percent)
                VALUES (?, ?, ?, ?)
            """, registered)

        conn.commit()
        conn.close()

    def get_registered_plates(self):
        """Retrieve list of all registered vehicle plate strings."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT plate_number FROM registered_vehicles")
        rows = cursor.fetchall()
        conn.close()
        return [r[0] for r in rows]

    def allocate_slot(self, category="REGULAR"):
        """Find the nearest available slot matching category."""
        conn = self.get_connection()
        cursor = conn.cursor()

        slot_pref = "VIP" if category == "VIP" else "GENERAL"
        
        # Try preferred slot
        cursor.execute("""
            SELECT slot_id FROM parking_slots 
            WHERE status = 'AVAILABLE' AND slot_type = ? 
            ORDER BY floor ASC, slot_id ASC LIMIT 1
        """, (slot_pref,))
        row = cursor.fetchone()

        # Fallback to any general slot
        if row is None:
            cursor.execute("""
                SELECT slot_id FROM parking_slots 
                WHERE status = 'AVAILABLE' 
                ORDER BY floor ASC, slot_id ASC LIMIT 1
            """)
            row = cursor.fetchone()

        conn.close()
        return row[0] if row else None

    def calculate_parking_fee(self, duration_minutes, discount_percent=0.0, mall_spent=0.0):
        """
        Mall Pricing Rules:
        - 0 to 15 mins: Free grace period ($0)
        - 1st hour: $40
        - 2nd to 4th hours: $30/hr
        - 5th hour onwards: $20/hr
        - Daily Max Cap: $300
        - Mall Discount: If shopping spent >= $500, first 2 hours are free ($70 credit)!
        - Registered VIP / Monthly: Applied per registration discount_percent
        """
        if duration_minutes <= 15:
            return 0.0, 0.0, 0.0

        hours = duration_minutes / 60.0
        base_fee = 0.0

        if hours <= 1.0:
            base_fee = 40.0
        elif hours <= 4.0:
            base_fee = 40.0 + (hours - 1.0) * 30.0
        else:
            base_fee = 40.0 + (3.0 * 30.0) + (hours - 4.0) * 20.0

        base_fee = min(base_fee, 300.0)

        # Mall shopping voucher discount
        mall_discount = 0.0
        if mall_spent >= 500.0:
            mall_discount = min(base_fee, 70.0)  # Free first 2 hours

        # Registration category discount
        sub_discount = (base_fee - mall_discount) * (discount_percent / 100.0)
        total_discount = mall_discount + sub_discount
        final_amount = max(0.0, base_fee - total_discount)

        return round(base_fee, 2), round(total_discount, 2), round(final_amount, 2)

    def process_entry(self, plate_number):
        """
        Record vehicle entry at the smart parking barrier.
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        # Check if vehicle is already inside
        cursor.execute("""
            SELECT log_id, entry_time, slot_id FROM parking_logs
            WHERE plate_number = ? AND status = 'ACTIVE'
        """, (plate_number,))
        existing = cursor.fetchone()
        if existing:
            conn.close()
            return {
                "success": False,
                "message": f"Vehicle {plate_number} is already parked at slot {existing[2]}!",
                "slot_id": existing[2],
                "entry_time": existing[1]
            }

        # Check registration category
        cursor.execute("SELECT owner_name, category FROM registered_vehicles WHERE plate_number = ?", (plate_number,))
        reg_info = cursor.fetchone()
        category = reg_info[1] if reg_info else "REGULAR"
        owner_name = reg_info[0] if reg_info else "Guest Customer"

        # Allocate slot
        slot_id = self.allocate_slot(category)
        if not slot_id:
            conn.close()
            return {
                "success": False,
                "message": "Parking Lot is FULL! Please wait for a vehicle to depart.",
                "slot_id": None
            }

        # Record entry
        entry_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            INSERT INTO parking_logs (plate_number, entry_time, slot_id, status)
            VALUES (?, ?, ?, 'ACTIVE')
        """, (plate_number, entry_time, slot_id))
        
        # Mark slot occupied
        cursor.execute("""
            UPDATE parking_slots SET status = 'OCCUPIED', current_plate = ? WHERE slot_id = ?
        """, (plate_number, slot_id))

        conn.commit()
        conn.close()

        return {
            "success": True,
            "action": "ENTRY",
            "plate_number": plate_number,
            "owner_name": owner_name,
            "category": category,
            "slot_id": slot_id,
            "entry_time": entry_time,
            "barrier": "OPEN",
            "message": f"Welcome! Slot {slot_id} assigned. Barrier opened."
        }

    def process_exit(self, plate_number, mall_spent=0.0, simulated_duration_mins=None, payment_method="FASTag"):
        """
        Record vehicle exit, calculate parking charges, trigger payment, and open gate barrier.
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT log_id, entry_time, slot_id FROM parking_logs
            WHERE plate_number = ? AND status = 'ACTIVE'
            ORDER BY log_id DESC LIMIT 1
        """, (plate_number,))
        active_log = cursor.fetchone()

        if not active_log:
            conn.close()
            return {
                "success": False,
                "message": f"No active entry found for vehicle {plate_number}."
            }

        log_id, entry_str, slot_id = active_log
        entry_time = datetime.strptime(entry_str, "%Y-%m-%d %H:%M:%S")
        
        # Duration calculation
        if simulated_duration_mins is not None:
            duration_minutes = simulated_duration_mins
            exit_time = entry_time + timedelta(minutes=duration_minutes)
        else:
            exit_time = datetime.now()
            duration_minutes = max(1.0, (exit_time - entry_time).total_seconds() / 60.0)

        exit_str = exit_time.strftime("%Y-%m-%d %H:%M:%S")

        # Get registration discount if any
        cursor.execute("SELECT category, discount_percent FROM registered_vehicles WHERE plate_number = ?", (plate_number,))
        reg = cursor.fetchone()
        discount_percent = reg[1] if reg else 0.0
        category = reg[0] if reg else "REGULAR"

        base_fee, total_disc, final_fee = self.calculate_parking_fee(
            duration_minutes, discount_percent=discount_percent, mall_spent=mall_spent
        )

        # Update log
        cursor.execute("""
            UPDATE parking_logs
            SET exit_time = ?, duration_minutes = ?, base_fee = ?, discount_amount = ?,
                final_amount = ?, payment_method = ?, payment_status = 'PAID', status = 'COMPLETED'
            WHERE log_id = ?
        """, (exit_str, round(duration_minutes, 1), base_fee, total_disc, final_fee, payment_method, log_id))

        # Free slot
        cursor.execute("""
            UPDATE parking_slots SET status = 'AVAILABLE', current_plate = NULL WHERE slot_id = ?
        """, (slot_id,))

        conn.commit()
        conn.close()

        receipt = {
            "success": True,
            "action": "EXIT",
            "log_id": log_id,
            "plate_number": plate_number,
            "category": category,
            "slot_id": slot_id,
            "entry_time": entry_str,
            "exit_time": exit_str,
            "duration_formatted": f"{int(duration_minutes // 60)}h {int(duration_minutes % 60)}m",
            "duration_minutes": round(duration_minutes, 1),
            "base_fee": base_fee,
            "discount_amount": total_disc,
            "final_amount": final_fee,
            "payment_method": payment_method,
            "payment_status": "PAID",
            "barrier": "OPEN",
            "message": f"Payment of ₹{final_fee:.2f} successful via {payment_method}. Thank you, visit again!"
        }
        return receipt

    def get_parking_occupancy(self):
        """Return total, occupied, and available slots count."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT status, COUNT(*) FROM parking_slots GROUP BY status")
        rows = dict(cursor.fetchall())
        conn.close()
        occupied = rows.get("OCCUPIED", 0)
        available = rows.get("AVAILABLE", 0)
        total = occupied + available
        return {"total": total, "occupied": occupied, "available": available}
